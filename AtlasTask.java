package com.atlas.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * A collection of tasks sharing a common deadline. Subgroups of tasks can have shorter deadlines.
 */
public class ReqHandlerTasks {
    private static final ThreadLocal<ReqHandlerTasks> ATLAS_TASKS = new ThreadLocal<>();
    private final List<ReqHandlerTasks> subtasks = new ArrayList<>();

    private final long timeoutMs;
    private final Executor executor;
    private final long startTimeMs;
    private final CompletionService<Object> completionService;
    private final BlockingQueue<Future<Object>> completionQueue = new LinkedBlockingQueue<>();
    private final Map<Future<Object>, Future<Object>> futureObjects = new HashMap<>();

    private boolean isShutdown = false;

    /**
     * Get the current tasks for this thread.
     */
    public static ReqHandlerTasks getInstance() {
        return ATLAS_TASKS.get();
    }
    /**
     * initializes the top level set of tasks.
     */
    public ReqHandlerTasks(Executor executor, long timeoutMs, HttpServletRequest request, HttpServletResponse response) {
        this.executor = executor;
        this.timeoutMs = timeoutMs;
        this.startTimeMs = System.currentTimeMillis();

        this.completionService = getCompletionService();
        ATLAS_TASKS.set(this);
    }
    /**
     * Create a group of subtasks with the same deadline as the current tasks.
     * A convenience method for creating a group of tasks with the same current deadline.
     */
    public ReqHandlerTasks() {
        this(100);
    }
    public ReqHandlerTasks(int timeoutPercentage) {
        this(getInstance(), timeoutPercentage);
    }
    /**
     * add to subtasks list with a shortened deadline relative to the specified tasks.
     */
    public ReqHandlerTasks(ReqHandlerTasks tasks, int timeoutPercentage) {
        this(tasks, tasks.timeoutMs * timeoutPercentage / 100, TimeUnit.MILLISECONDS);
    }

    public ReqHandlerTasks(ReqHandlerTasks tasks, long timeout, TimeUnit unit) {
        this.timeoutMs = Math.min(tasks.timeoutMs, TimeUnit.MILLISECONDS.convert(timeout, unit));
        this.executor = tasks.executor;
        this.startTimeMs = tasks.startTimeMs;

        this.completionService = getCompletionService();
        tasks.subtasks.add(this);
    }

    /**
     * Complete all tasks and subtasks.
     */
    public static void completeAll() {
        getInstance().complete();
    }

    /**
     * Submit a new task to the current set of tasks.
     */
    public static <T> AsyncResponse<T> submit(Callable<T> service) {
        return getInstance().add(service);
    }

    /**
     * Submit a new task with a shorter deadline.
     * This is a convenience method for creating a new set of tasks
     * containing a single task with a shortened deadline.
     */
    public static <T> AsyncResponse<T> submit(Callable<T> service, int timeoutPercentage) {
        return getInstance().add(service, timeoutPercentage);
    }

    /**
     * Submit a new task with a shorter deadline.
     * This is a convenience method for creating a new set of tasks
     * containing a single task with a shortened deadline.
     */
    public static <T> AsyncResponse<T> submit(Callable<T> service, long timeout, TimeUnit unit) {
        return getInstance().add(service, timeout, unit);
    }

    /**
     * Submit a new task with a shorter deadline relative to this set's deadline.
     * This is a convenience method for creating a new set
     * containing a single task with the shortened deadline relative to this set.
     */
    public <T> AsyncResponse<T> add(Callable<T> service) {
        return add(service, 100);
    }
    public <T> AsyncResponse<T> add(Callable<T> service, int timeoutPercentage) {
        return add(service, timeoutMs * timeoutPercentage / 100, TimeUnit.MILLISECONDS);
    }

    /**
     * Submit with completionService.submit(), futre.get() block on completionFuture.call(). 
     */
    public <T> AsyncResponse<T> add(Callable<T> nonFinalService, long timeout, TimeUnit unit) {
        synchronized (this) {
            if (isShutdown) {
                RuntimeException e = new IllegalStateException("shutdown");
                LOGGER.warn("[ZOMBIE TASK]", e);
                throw e;
            }
            // Capture performance information for the service call
            final Callable<T> service;
            service = nonFinalService;

            // Run the service asynchronously, returning an async response that the
            // caller can use to get the results of the call.
            final ReqHandlerTasks tasks = new ReqHandlerTasks(this, timeout, unit);
            final BeanContainer beanContainer = BeanContainer.getInstance();
            final Future<Object> completionServiceFuture = completionService.submit(
                    TransactionLogger.wrapCallable(
                            new Callable<Object>() {
                                @Override
                                public Object call() throws Exception {
                                    ATLAS_TASKS.set(tasks);
                                    BeanContainer.initialize(beanContainer);
                                    long startTimeMillis = System.currentTimeMillis();
                                    Object result = service.call();
                                    long endTimeMillis = System.currentTimeMillis();
                                    LOGGER.debug("[BENCH]",
                                            LogVar.with("name", service.getClass().getName()),
                                            LogVar.with("elapsedTimeMs", endTimeMillis - startTimeMillis),
                                            LogVar.with("startTimeMs", startTimeMillis),
                                            LogVar.with("endTimeMs", endTimeMillis));
                                    return result;
                                }
                            }, "NESTED_TX", nonFinalService.toString()
                    )
            );
            Future<Object> atlasTasksFuture = new Future<Object>() {
                @Override
                public boolean cancel(boolean mayInterruptIfRunning) {
                    boolean isCancelled = completionServiceFuture.cancel(mayInterruptIfRunning);
                    if (isCancelled) {
                        LOGGER.warn("[CANCELLED]",
                                LogVar.with("serviceName", service.toString()));
                    }
                    return isCancelled;
                }

                @Override
                public boolean isCancelled() {
                    return completionServiceFuture.isCancelled();
                }

                @Override
                public boolean isDone() {
                    return completionServiceFuture.isDone();
                }

                @Override
                public Object get() throws InterruptedException, ExecutionException {
                    try {
                        return completionServiceFuture.get();
                    } catch (InterruptedException originalException) {
                        InterruptedException newException = new InterruptedException(service.toString());
                        newException.setStackTrace(originalException.getStackTrace());
                        throw newException;
                    } catch (ExecutionException originalException) {
                        ExecutionException newException =
                                new ExecutionException(service.toString(), originalException.getCause());
                        newException.setStackTrace(originalException.getStackTrace());
                        throw newException;
                    }
                }

                @Override
                public Object get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
                    try {
                        return completionServiceFuture.get(timeout, unit);
                    } catch (InterruptedException originalException) {
                        InterruptedException newException = new InterruptedException(service.toString());
                        newException.setStackTrace(originalException.getStackTrace());
                        throw newException;
                    } catch (ExecutionException originalException) {
                        ExecutionException newException =
                                new ExecutionException(service.toString(), originalException.getCause());
                        newException.setStackTrace(originalException.getStackTrace());
                        throw newException;
                    } catch (TimeoutException originalException) {
                        TimeoutException newException = new TimeoutException(service.toString());
                        newException.setStackTrace(originalException.getStackTrace());
                        LOGGER.error("Timed out getting result of asynchronous call",
                                LogVar.with("serviceName", service.toString()),
                                LogVar.with("timeoutMs", tasks.timeoutMs),
                                newException);
                        throw newException;
                    }
                }

                @Override
                public String toString() {
                    return super.toString() + "(" + service.toString() + ")";
                }
            };
            futureObjects.put(completionServiceFuture, atlasTasksFuture);
            return newAsyncResponse(atlasTasksFuture, tasks.getDeadlineMs());
        }
    }

    /**
     * Poll for completed tasks until there are no more or the deadline is reached.
     */
    public void complete() {
        // don't want to lock this set of tasks for an extended period
        // so at the risk of accuracy make a copy of its subtasks and test that
        ReqHandlerTasks[] subtasks;
        synchronized (this) {
            subtasks = !this.subtasks.isEmpty() ?
                    this.subtasks.toArray(new ReqHandlerTasks[this.subtasks.size()]) : null;
        }
        if (subtasks != null) {
            for (ReqHandlerTasks tasks : subtasks) {
                tasks.complete();
            }
        }
        while (hasNext()) {
            long timeout = getTimeoutMs();
            try {
                completionService.poll(timeout, TimeUnit.MILLISECONDS);
            } catch (Exception ignored) {
                LOGGER.debug("unexpected exception", ignored);
            }
        }
    }

    /**
     * Get the deadline for these tasks.
     */
    public long getDeadlineMs() {
        return startTimeMs + timeoutMs;
    }

    /**
     * Are all the submitted tasks done and their results retrieved via this iteration?
     */
    public boolean hasNext() {
        long timeout = getTimeoutMs();
        if (timeout <= 0) {
            // once the deadline has passed, the queue is treated as empty
            return false;
        }
        synchronized (this) {
            if (!completionQueue.isEmpty()) {
                return true;
            }
            for (Future<Object> futureObject : futureObjects.values()) {
                if (!futureObject.isDone()) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Get the result of the next completed task or <code>null</code> past the deadline.
     */
    public Object next() throws Exception {
        long timeout = getTimeoutMs();
        if (timeout <= 0) {
            return null;
        }
        Future<Object> completionServiceFuture = completionService.poll(timeout, TimeUnit.MILLISECONDS);
        Future<Object> atlasTasksFuture = futureObjects.get(completionServiceFuture);
        try {
            return atlasTasksFuture != null ? atlasTasksFuture.get() : null;
        } catch (ExecutionException e) {
            throw e.getCause() != null && e.getCause() instanceof Exception ? (Exception) e.getCause() : e;
        }
    }

    /**
     * Shutdown the completion service by preventing additional tasks from being submitted,
     * shutting down any subtasks, and canceling all the current tasks.
     */
    public void shutdown() {
        if (isShutdown) {
            return;
        }
        synchronized (this) {
            if (!isShutdown) {
                isShutdown = true;
                for (ReqHandlerTasks tasks : subtasks) {
                    tasks.shutdown();
                }
                for (Future<Object> futureObject : futureObjects.values()) {
                    futureObject.cancel(true);
                }
            }
        }
    }

    /**
     * Shared mechanism among all instances for creating a {@link java.util.concurrent.Executor}.
     */
    private ExecutorCompletionService<Object> getCompletionService() {
        return new ExecutorCompletionService<>(executor, completionQueue);
    }

    /**
     * Add up the start time plus the timeout time to get the deadline and subtract the current time to obtain the
     * timeout interval for a poll call. If the interval is less than zero then the deadline has passed.
     *
     * @return a {@code long} containing the timeout interval based on the current time
     */
    private long getTimeoutMs() {
        return startTimeMs + timeoutMs - System.currentTimeMillis();
    }

    /**
     * Create a typed {@link com.atlas.core.AsyncResponse} from an untyped {@link java.util.concurrent.Future}.
     */
    @SuppressWarnings("unchecked")
    private <T> AsyncResponse<T> newAsyncResponse(Future<Object> futureObject, long deadlineMs) {
        return new AsyncResponse<>((Future<T>) futureObject, deadlineMs);
    }

    private <T> invokeAll () {
        ExecutorService executor = Executors.newWorkStealingPool();

        List<Callable<String>> callables = Arrays.asList(
                () -> "task1",
                () -> "task2",
                () -> "task3");

        executor.invokeAll(callables)
            .stream()
            .map(future -> {
                try {
                    return future.get();
                }
                catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            })
            .forEach(System.out::println);
    }

    private <T> scheduleWithFixedDelay () {
        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);

        Runnable task = () -> {
            try {
                TimeUnit.SECONDS.sleep(2);
                System.out.println("Scheduling: " + System.nanoTime());
            }
            catch (InterruptedException e) {
                System.err.println("task interrupted");
            }
        };

        executor.scheduleWithFixedDelay(task, 0, 1, TimeUnit.SECONDS);
    }
}

public final class BeanContainer {
    /**
     * A marker interface indicating the class is intended to be maintained in the bean container.
     */
    public interface Bean {
    }

    /**
     * The bean factory when the bean is not found in the bean container.
     *
     * @param <T> the type of bean.
     */
    public interface BeanFactory<T extends Bean> {
        T newInstance();
    }

    /**
     * The {@link java.lang.ThreadLocal} instance of the bean container.
     */
    private static final ThreadLocal<BeanContainer> INSTANCE = new ThreadLocal<>();

    /**
     * The beans are maintained in a map with {@link org.codehaus.jackson.type.TypeReference} keys and
     * {@link com.walmart.atlas.core.BeanContainer.Bean} implementing instances.
     */
    final Map<TypeReference, Object> beans = new ConcurrentHashMap<>();

    /**
     * The current {@link javax.servlet.http.HttpServletRequest request}
     */
    private final HttpServletRequest request;

    /**
     * The current {@link javax.servlet.http.HttpServletResponse response}
     */
    private final HttpServletResponse response;

    /**
     * A group of shared tasks with the maximum deadline for the current request.
     */
    private final ReqHandlerTasks sharedTasks;

    /**
     * The construct of the share bean container for the current request.
     */
    protected BeanContainer(HttpServletRequest request, HttpServletResponse response) {
        this.request = request;
        this.response = response;
        this.sharedTasks = AtlasTasks.getInstance();
    }

    /**
     * Retrieve a bean or construct it if not found.
     *
     * @param beanType    the bean's {@link org.codehaus.jackson.type.TypeReference}
     * @param beanFactory the bean's factory when not present in the bean container
     * @param <T>         the type of bean.
     * @return the bean instance for the current request
     */
    public static <T extends Bean> T getBean(TypeReference<T> beanType, BeanFactory<T> beanFactory) {
        T bean = getBean(beanType);
        if (bean == null) {
            T bean0 = beanFactory.newInstance();
            bean = putBean(beanType, bean0);
            if (bean == null) {
                bean = bean0;
            }
        }
        return bean;
    }

    /**
     * Get the shared bean container instance for the current request.
     */
    @SuppressWarnings("UnnecessaryLocalVariable")
    public static BeanContainer getInstance() {
        BeanContainer instance = INSTANCE.get();
        return instance;
    }

    /**
     * Get the current request.
     */
    // TODO make this package private
    public static HttpServletRequest getRequest() {
        return getInstance().getRequestInternal();
    }

    /**
     * Get the current response.
     */
    // TODO make this package private
    public static HttpServletResponse getResponse() {
        return getInstance().getResponseInternal();
    }

    /**
     * Initialize the container, useful for unit testing.
     *
     * @param request  the current {@link javax.servlet.http.HttpServletRequest}
     * @param response the current {@link javax.servlet.http.HttpServletResponse}
     */
    public static void initialize(HttpServletRequest request, HttpServletResponse response) {
        initialize(new BeanContainer(request, response));
    }

    /**
     * Is the bean container initialized?
     */
    public static boolean isInitialized() {
        return BeanContainer.getInstance() != null && BeanContainer.getRequest() != null;
    }

    /**
     * Submit a shared task.
     */
    // TODO make this package private
    public static <T> AsyncResponse<T> submitSharedTask(Callable<T> service, int timeoutPercentage) {
        return getInstance().sharedTasks.add(service, timeoutPercentage);
    }

    private <T> AsyncResponse<T> submitTask(Callable<T> service, int timeoutPercentage) {
        return sharedTasks.add(service, timeoutPercentage);
    }

    /**
     * Submit a shared task.
     */
    // TODO make this package private
    public static <T> AsyncResponse<T> submitSharedTask(Callable<T> service, long timeout, TimeUnit timeUnit) {
        return getInstance().submitTask(service, timeout, timeUnit);
    }

    private <T> AsyncResponse<T> submitTask(Callable<T> service, long timeout, TimeUnit timeUnit) {
        return sharedTasks.add(service, timeout, timeUnit);
    }

    /**
     * Submit a shared task.
     */
    // TODO make this package private
    public static <T> AsyncResponse<T> submitSharedTask(Callable<T> service) {
        return getInstance().submitTask(service);
    }

    private <T> AsyncResponse<T> submitTask(Callable<T> service) {
        return sharedTasks.add(service);
    }

    /**
     * Put a bean into the container.
     */
    // allow unit tests
    @SuppressWarnings("unchecked")
    protected static <T extends Bean> T putBean(TypeReference<T> beanType, T bean) {
        return (T) getInstance().beans.putIfAbsent(beanType, bean);
    }

    /**
     * Initialize the bean container on the current thread.
     *
     * @param BeanContainer the shared bean container from the spawning thread
     */
    static void initialize(BeanContainer BeanContainer) {
        if (BeanContainer != null) {
            INSTANCE.set(BeanContainer);
        } else {
            INSTANCE.remove();
        }
    }

    /**
     * Get a bean out of the container.
     */
    @SuppressWarnings("unchecked")
    private static <T> T getBean(TypeReference<T> beanType) {
        return (T) getInstance().beans.get(beanType);
    }

    /**
     * Get the current request.
     */
    HttpServletRequest getRequestInternal() {
        return request;
    }

    /**
     * Get the current response.
     */
    HttpServletResponse getResponseInternal() {
        return response;
    }
}

