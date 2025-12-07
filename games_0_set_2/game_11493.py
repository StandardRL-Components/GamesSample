import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:06:50.054883
# Source Brief: brief_01493.md
# Brief Index: 1493
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls three bouncing balls.
    The goal is to hit targets to score 100 points within a time limit.
    Synchronizing hits with all three balls provides a score multiplier.
    Difficulty increases as the score gets higher.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three bouncing balls to hit targets and score points. "
        "Synchronize hits for a score multiplier before time runs out."
    )
    user_guide = "Controls: Use ← and → arrow keys to apply horizontal force to all three balls."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_SCORE = 100
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (40, 40, 55)
    COLOR_TARGET = (255, 220, 0)
    COLOR_OBSTACLE = (80, 80, 100)
    COLOR_TEXT = (240, 240, 240)
    BALL_COLORS = [(255, 80, 80), (80, 255, 80), (80, 120, 255)] # R, G, B

    # Physics
    GRAVITY = 600.0  # pixels/sec^2
    BOUNCE_DAMPENING = 0.9
    INITIAL_BALL_SPEED = 180.0  # pixels/sec
    BALL_RADIUS = 12
    TARGET_SIZE = (40, 40)
    
    # Gameplay
    MULTIPLIER_WINDOW_SEC = 0.2
    TARGET_RESPAWN_SEC = 10.0
    MAX_OBSTACLES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # This will be set properly in reset()
        self.dt = 1.0 / 30.0 # Assume 30 FPS for physics calculations
        self.multiplier_window_steps = int(self.MULTIPLIER_WINDOW_SEC / self.dt)
        self.target_respawn_steps = int(self.TARGET_RESPAWN_SEC / self.dt)

        self.balls = []
        self.targets = []
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.last_hit_steps = {}
        self.last_target_hit_step = 0
        self.screen_flash_timer = 0

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.last_target_hit_step = 0
        self.screen_flash_timer = 0

        self.balls = self._initialize_balls()
        self.obstacles = []
        self.targets = [self._spawn_target() for _ in range(3)]
        self.particles = []
        
        self.last_hit_steps = {i: -self.multiplier_window_steps * 2 for i in range(len(self.balls))}

        return self._get_observation(), self._get_info()

    def _initialize_balls(self):
        balls = []
        for i in range(3):
            balls.append({
                "pos": np.array([
                    self.SCREEN_WIDTH * (i + 1) / 4.0,
                    self.SCREEN_HEIGHT / 3.0
                ]),
                "vel": np.array([
                    self.np_random.choice([-1.0, 1.0]) * self.INITIAL_BALL_SPEED * 0.5,
                    0.0
                ]),
                "color": self.BALL_COLORS[i],
                "id": i
            })
        return balls

    def _spawn_target(self):
        while True:
            pos_x = self.np_random.integers(self.BALL_RADIUS, self.SCREEN_WIDTH - self.TARGET_SIZE[0] - self.BALL_RADIUS)
            pos_y = self.np_random.integers(self.BALL_RADIUS, self.SCREEN_HEIGHT - self.TARGET_SIZE[1] - self.BALL_RADIUS)
            new_target = pygame.Rect(pos_x, pos_y, self.TARGET_SIZE[0], self.TARGET_SIZE[1])
            
            # Ensure it doesn't overlap with existing targets or obstacles
            if not any(new_target.colliderect(t) for t in self.targets) and \
               not any(new_target.colliderect(o) for o in self.obstacles):
                return new_target

    def _spawn_obstacle(self):
        width = self.np_random.integers(60, 120)
        height = self.np_random.integers(15, 25)
        while True:
            pos_x = self.np_random.integers(0, self.SCREEN_WIDTH - width)
            pos_y = self.np_random.integers(100, self.SCREEN_HEIGHT - height - 100)
            new_obstacle = pygame.Rect(pos_x, pos_y, width, height)
            
            # Avoid overlap with targets
            if not any(new_obstacle.colliderect(t) for t in self.targets):
                self.obstacles.append(new_obstacle)
                # SFX: Obstacle spawn sound
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        self._handle_input(movement)
        reward += self._update_physics_and_collisions()
        self._update_particles()
        self._update_progression()
        
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1

        self.steps += 1
        
        # Anti-softlock mechanism
        if self.steps - self.last_target_hit_step > self.target_respawn_steps:
            self.targets = [self._spawn_target() for _ in range(3)]
            self.last_target_hit_step = self.steps
            # SFX: Target respawn woosh

        terminated = self._check_termination()
        truncated = False # No truncation condition other than time limit
        if terminated:
            if self.score >= self.MAX_SCORE:
                reward += 100  # Victory reward
            else:
                reward -= 10  # Timeout penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3:  # Left
            for ball in self.balls:
                ball["vel"][0] = -self.ball_speed
        elif movement == 4:  # Right
            for ball in self.balls:
                ball["vel"][0] = self.ball_speed

    def _update_physics_and_collisions(self):
        step_reward = 0
        
        for ball in self.balls:
            # Update velocity with gravity
            ball["vel"][1] += self.GRAVITY * self.dt
            
            # Update position
            ball["pos"] += ball["vel"] * self.dt

            # Wall collisions
            if ball["pos"][0] < self.BALL_RADIUS:
                ball["pos"][0] = self.BALL_RADIUS
                ball["vel"][0] *= -self.BOUNCE_DAMPENING
                # SFX: Bounce
            elif ball["pos"][0] > self.SCREEN_WIDTH - self.BALL_RADIUS:
                ball["pos"][0] = self.SCREEN_WIDTH - self.BALL_RADIUS
                ball["vel"][0] *= -self.BOUNCE_DAMPENING
                # SFX: Bounce

            if ball["pos"][1] < self.BALL_RADIUS:
                ball["pos"][1] = self.BALL_RADIUS
                ball["vel"][1] *= -self.BOUNCE_DAMPENING
                # SFX: Bounce
            elif ball["pos"][1] > self.SCREEN_HEIGHT - self.BALL_RADIUS:
                ball["pos"][1] = self.SCREEN_HEIGHT - self.BALL_RADIUS
                ball["vel"][1] *= -self.BOUNCE_DAMPENING
                # SFX: Bounce

            # Obstacle collisions
            ball_rect = pygame.Rect(ball["pos"][0] - self.BALL_RADIUS, ball["pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            for obstacle in self.obstacles:
                if ball_rect.colliderect(obstacle):
                    # Simple axis-aligned bounce logic
                    overlap_x = (self.BALL_RADIUS + obstacle.width / 2) - abs(ball["pos"][0] - obstacle.centerx)
                    overlap_y = (self.BALL_RADIUS + obstacle.height / 2) - abs(ball["pos"][1] - obstacle.centery)

                    if overlap_x < overlap_y:
                        ball["vel"][0] *= -self.BOUNCE_DAMPENING
                        ball["pos"][0] += np.sign(ball["pos"][0] - obstacle.centerx) * overlap_x
                    else:
                        ball["vel"][1] *= -self.BOUNCE_DAMPENING
                        ball["pos"][1] += np.sign(ball["pos"][1] - obstacle.centery) * overlap_y
                    # SFX: Bounce_hard

            # Target collisions
            for i, target in enumerate(self.targets):
                if target.collidepoint(ball["pos"]):
                    step_reward += 0.1
                    self.score += 1
                    self.last_target_hit_step = self.steps
                    self.last_hit_steps[ball["id"]] = self.steps
                    
                    self._create_particles(ball["pos"], self.COLOR_TARGET, 30)
                    self.targets[i] = self._spawn_target()
                    # SFX: Target hit
                    
                    # Check for multiplier
                    hit_times = list(self.last_hit_steps.values())
                    if max(hit_times) - min(hit_times) <= self.multiplier_window_steps:
                        step_reward += 1.0
                        self.score += 5 # Bonus points
                        self.screen_flash_timer = 3 # frames
                        self._create_particles(ball["pos"], (255,255,255), 50)
                        # SFX: Multiplier success
                        # Reset hit times to prevent re-triggering
                        self.last_hit_steps = {i: -self.multiplier_window_steps * 2 for i in range(len(self.balls))}

        return step_reward

    def _update_progression(self):
        # Increase ball speed every 10 points
        current_speed_level = self.score // 10
        self.ball_speed = self.INITIAL_BALL_SPEED * (1 + 0.05 * current_speed_level)
        
        # Add obstacles every 20 points
        num_obstacles_to_add = (self.score // 20) - len(self.obstacles)
        if num_obstacles_to_add > 0 and len(self.obstacles) < self.MAX_OBSTACLES:
            for _ in range(num_obstacles_to_add):
                if len(self.obstacles) < self.MAX_OBSTACLES:
                    self._spawn_obstacle()

    def _check_termination(self):
        return self.score >= self.MAX_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = 100 * (self.screen_flash_timer / 3)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle, border_radius=3)
        
        # Targets
        for target in self.targets:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, target, border_radius=5)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = p['color'] + (int(alpha),)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Balls
        for ball in self.balls:
            pos_int = (int(ball["pos"][0]), int(ball["pos"][1]))
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            glow_color = ball["color"] + (50,)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))
            
            # Main ball
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, ball["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, ball["color"])
    
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:03}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        timer_width = (self.SCREEN_WIDTH / 3) * time_ratio
        timer_rect = pygame.Rect(self.SCREEN_WIDTH - 20 - timer_width, 22, timer_width, 20)
        
        hue = 120 * time_ratio # Green to Red
        timer_color = pygame.Color(0)
        timer_color.hsva = (hue, 100, 80, 100)
        pygame.draw.rect(self.screen, timer_color, timer_rect, border_radius=5)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.MAX_SCORE:
                msg = "VICTORY!"
            else:
                msg = "TIME UP"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 200)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.uniform(0.3, 0.8) # seconds
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel'] * self.dt
            p['vel'] *= 0.95 # Damping
            p['life'] -= self.dt
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # In this mode, we need a display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bounce Sync")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4
        
        action = [movement_action, 0, 0] # space/shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()