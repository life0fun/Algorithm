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

import com.atlas.core.config.DebugModeConfig;
import com.atlas.core.perf.PagePerformanceInfo;
import com.atlas.core.perf.ServiceCallPerformanceCaptor;
import com.platform.logging.client.type.TransactionLogger;
import com.platform.logmon.LogVar;

/**
 * A collection of tasks sharing a common deadline. Subgroups of tasks can have shorter deadlines.
 */
public class AtlasTasks {
    private static final ThreadLocal<AtlasTasks> ATLAS_TASKS = new ThreadLocal<>();
    private static final Logger LOGGER = LoggerFactory.getLogger(AtlasTasks.class);

    final PagePerformanceInfo mDebugInfo;

    private final long timeoutMs;
    private final Executor executor;
    private final long startTimeMs;
    private final CompletionService<Object> completionService;
    private final BlockingQueue<Future<Object>> completionQueue = new LinkedBlockingQueue<>();
    private final List<AtlasTasks> subtasks = new ArrayList<>();
    private final Map<Future<Object>, Future<Object>> futureObjects = new HashMap<>();

    private boolean isShutdown = false;

    /**
     * Create subtasks with a shortened deadline relative to the specified tasks.
     */
    public AtlasTasks(AtlasTasks tasks, int timeoutPercentage) {
        this(tasks, tasks.timeoutMs * timeoutPercentage / 100, TimeUnit.MILLISECONDS);
    }

    public AtlasTasks(AtlasTasks tasks, long timeout, TimeUnit unit) {
        this.timeoutMs = Math.min(tasks.timeoutMs, TimeUnit.MILLISECONDS.convert(timeout, unit));
        this.executor = tasks.executor;
        this.mDebugInfo = tasks.mDebugInfo;
        this.startTimeMs = tasks.startTimeMs;

        this.completionService = getCompletionService();
        tasks.subtasks.add(this);
    }

    /**
     * Create subtasks with a shortened deadline relative to the current tasks.
     */
    public AtlasTasks(int timeoutPercentage) {
        this(getInstance(), timeoutPercentage);
    }

    /**
     * Create a group of subtasks with the same deadline as the current tasks.
     * A convenience method for creating a group of tasks with the same current deadline.
     */
    public AtlasTasks() {
        this(100);
    }

    /**
     * Called from the {@link com.atlas.core.AtlasThreadContextFilter};
     * initializes the top level set of tasks.
     */
    public AtlasTasks(Executor executor, long timeoutMs, HttpServletRequest request, HttpServletResponse response) {
        this.executor = executor;
        this.timeoutMs = timeoutMs;
        this.mDebugInfo = DebugModeConfig.debugModeEnabled ? new PagePerformanceInfo() : null;
        this.startTimeMs = System.currentTimeMillis();

        this.completionService = getCompletionService();
        ATLAS_TASKS.set(this);
        CustomerLocationAndStores.newCustomerLocationAndStores(request, response);
    }

    /**
     * Complete all tasks and subtasks.
     */
    public static void completeAll() {
        getInstance().complete();
    }

    /**
     * Get the current tasks for this thread.
     */
    public static AtlasTasks getInstance() {
        return ATLAS_TASKS.get();
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
    public <T> AsyncResponse<T> add(Callable<T> service, int timeoutPercentage) {
        return add(service, timeoutMs * timeoutPercentage / 100, TimeUnit.MILLISECONDS);
    }

    /**
     * Submit a new task with a shorter deadline relative to this set's deadline.
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
            if (mDebugInfo != null) {
                ServiceCallPerformanceCaptor<T> captor = new ServiceCallPerformanceCaptor<>(nonFinalService);
                mDebugInfo.registerServiceCall(captor);
                service = captor;
            } else {
                service = nonFinalService;
            }
            // Run the service asynchronously, returning an async response that the
            // caller can use to get the results of the call.
            final AtlasTasks tasks = new AtlasTasks(this, timeout, unit);
            final AtlasBeanContainer atlasBeanContainer = AtlasBeanContainer.getInstance();
            final CustomerLocationAndStores customerLocationAndStores = CustomerLocationAndStores.INSTANCE.get();
            final Future<Object> completionServiceFuture = completionService.submit(
                    TransactionLogger.wrapCallable(
                            new Callable<Object>() {
                                @Override
                                public Object call() throws Exception {
                                    ATLAS_TASKS.set(tasks);
                                    AtlasBeanContainer.initialize(atlasBeanContainer);
                                    CustomerLocationAndStores.INSTANCE.set(customerLocationAndStores);
                                    long startTimeMillis = System.currentTimeMillis();
                                    Object result = service.call();
                                    if (LOGGER.isDebugEnabled()) {
                                        long endTimeMillis = System.currentTimeMillis();
                                        LOGGER.debug("[BENCH]",
                                                LogVar.with("name", service.getClass().getName()),
                                                LogVar.with("elapsedTimeMs", endTimeMillis - startTimeMillis),
                                                LogVar.with("startTimeMs", startTimeMillis),
                                                LogVar.with("endTimeMs", endTimeMillis));
                                    }
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
     * Add a task to the set.
     */
    public <T> AsyncResponse<T> add(Callable<T> service) {
        return add(service, 100);
    }

    /**
     * Poll for completed tasks until there are no more or the deadline is reached.
     */
    public void complete() {
        // don't want to lock this set of tasks for an extended period
        // so at the risk of accuracy make a copy of its subtasks and test that
        AtlasTasks[] subtasks;
        synchronized (this) {
            subtasks = !this.subtasks.isEmpty() ?
                    this.subtasks.toArray(new AtlasTasks[this.subtasks.size()]) : null;
        }
        if (subtasks != null) {
            for (AtlasTasks tasks : subtasks) {
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
                for (AtlasTasks tasks : subtasks) {
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
}
