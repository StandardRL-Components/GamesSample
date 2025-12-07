import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:35:28.299901
# Source Brief: brief_01079.md
# Brief Index: 1079
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to reach a target velocity by applying impulses on wall bounces
    while fighting against speed decay.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a bouncing ball and apply directional impulses on wall collisions "
        "to reach a target velocity before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to apply an impulse on the next wall bounce "
        "in that direction."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed frame rate for physics
        self.TARGET_SPEED = 500
        self.MAX_TIME = 60.0
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS)

        # Physics & Gameplay
        self.BALL_RADIUS = 12
        self.INITIAL_SPEED = 80
        self.BOUNCE_ACCEL = 25  # Flat speed increase per bounce
        self.IMPULSE_STRENGTH = 150 # Speed added by player action
        self.TRAIL_LENGTH = 20
        self.SPARK_LIFETIME = 15 # frames

        # Visuals
        self.COLOR_BG = (10, 10, 26)
        self.COLOR_WALLS = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_BALL_SLOW = np.array([0, 120, 255])
        self.COLOR_BALL_FAST = np.array([255, 50, 50])
        self.COLOR_SPARK = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_impact = pygame.font.Font(None, 28)

        # --- Persistent State (survives reset) ---
        self.speed_decay_percentage = 0.05
        
        # --- Game State Variables (reset each episode) ---
        # These are initialized here to None and properly set in reset()
        self.steps = None
        self.score = None
        self.time_elapsed = None
        self.ball_pos = None
        self.ball_vel = None
        self.last_speed = None
        self.bounce_impulse = None
        self.trail = None
        self.sparks = None
        self.last_collision_info = None
        
        # self.reset() is called by the wrapper/runner, not needed here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.time_elapsed = 0.0
        
        # Ball state
        self.ball_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        angle = self.np_random.uniform(0, 2 * math.pi)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.INITIAL_SPEED
        self.last_speed = self.INITIAL_SPEED

        # Effects and input state
        self.bounce_impulse = None
        self.trail = []
        self.sparks = []
        self.last_collision_info = None # (text, pos, lifetime)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused
        
        self._handle_input(movement)
        self._update_physics()
        self._update_effects()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = False
        if terminated:
            self._handle_termination(self.get_speed() >= self.TARGET_SPEED)
        
        self.steps += 1
        self.time_elapsed += 1.0 / self.FPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.bounce_impulse = np.array([0, -1.0])
        elif movement == 2: # Down
            self.bounce_impulse = np.array([0, 1.0])
        elif movement == 3: # Left
            self.bounce_impulse = np.array([-1.0, 0])
        elif movement == 4: # Right
            self.bounce_impulse = np.array([1.0, 0])
        else: # No-op
            self.bounce_impulse = None

    def _update_physics(self):
        dt = 1.0 / self.FPS
        self.ball_pos += self.ball_vel * dt
        self.last_reward_from_collision = 0

        collided = False
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self._handle_collision('x', -1)
            collided = True
        elif self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self._handle_collision('x', 1)
            collided = True

        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self._handle_collision('y', -1)
            collided = True
        elif self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.HEIGHT - self.BALL_RADIUS
            self._handle_collision('y', 1)
            collided = True
        
        if collided:
            self.last_reward_from_collision = -1.0

    def _handle_collision(self, axis, side):
        # 1. Reflect velocity
        if axis == 'x':
            self.ball_vel[0] *= -1
        else: # 'y'
            self.ball_vel[1] *= -1

        # 2. Calculate current speed and apply decay
        speed = self.get_speed()
        speed *= (1 - self.speed_decay_percentage)
        
        # 3. Apply base acceleration
        speed += self.BOUNCE_ACCEL
        
        # 4. Apply player impulse if available
        if self.bounce_impulse is not None:
            # Add impulse vector to velocity
            self.ball_vel += self.bounce_impulse * self.IMPULSE_STRENGTH
            # Recalculate speed after impulse
            speed = self.get_speed()
            # Clear impulse for next bounce
            self.bounce_impulse = None
            # Play sound effect (placeholder)
            # print("SFX: Impulse Bounce")

        # 5. Re-normalize velocity to the new speed
        current_vel_norm = np.linalg.norm(self.ball_vel)
        if current_vel_norm > 1e-6:
            self.ball_vel = (self.ball_vel / current_vel_norm) * speed
        
        # 6. Create visual/UI feedback
        self._create_collision_effects(axis, side)
        # Play sound effect (placeholder)
        # print("SFX: Wall Hit")

    def _create_collision_effects(self, axis, side):
        # Create sparks
        collision_point = self.ball_pos.copy()
        if axis == 'x':
            collision_point[0] = self.BALL_RADIUS if side == -1 else self.WIDTH - self.BALL_RADIUS
        else:
            collision_point[1] = self.BALL_RADIUS if side == -1 else self.HEIGHT - self.BALL_RADIUS

        for _ in range(10):
            angle = self.np_random.uniform(math.pi * (axis == 'y'), math.pi + math.pi * (axis == 'y'))
            if side == 1: angle += math.pi
            speed = self.np_random.uniform(20, 80)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.sparks.append({'pos': collision_point.copy(), 'vel': vel, 'life': self.SPARK_LIFETIME})
        
        # Create impact text
        text = f"-{self.speed_decay_percentage:.0%}"
        text_pos = collision_point.copy()
        text_pos[0] = np.clip(text_pos[0], 40, self.WIDTH - 40)
        text_pos[1] = np.clip(text_pos[1], 20, self.HEIGHT - 20)
        self.last_collision_info = {'text': text, 'pos': text_pos, 'life': self.FPS * 1.5}


    def _update_effects(self):
        # Update trail
        self.trail.append(self.ball_pos.copy())
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)

        # Update sparks
        self.sparks = [s for s in self.sparks if s['life'] > 0]
        for spark in self.sparks:
            spark['pos'] += spark['vel'] * (1.0 / self.FPS)
            spark['life'] -= 1
        
        # Update collision text
        if self.last_collision_info:
            self.last_collision_info['life'] -= 1
            if self.last_collision_info['life'] <= 0:
                self.last_collision_info = None

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for speed increase
        current_speed = self.get_speed()
        speed_diff = current_speed - self.last_speed
        if speed_diff > 0:
            reward += 0.1 * math.floor(speed_diff / 10)
        self.last_speed = current_speed
        
        # Event-based reward for collision
        reward += self.last_reward_from_collision
        
        # Goal-oriented rewards (applied at termination)
        if self._check_termination():
            if current_speed >= self.TARGET_SPEED:
                reward += 100  # Win
            else:
                reward -= 100  # Timeout
        
        return reward

    def _check_termination(self):
        return (
            self.get_speed() >= self.TARGET_SPEED or
            self.time_elapsed >= self.MAX_TIME
        )

    def _handle_termination(self, won):
        if won:
            # On win, reset decay to base value
            self.speed_decay_percentage = 0.05
        else:
            # On loss, increase decay for next attempt, capped at 95%
            self.speed_decay_percentage = min(0.95, self.speed_decay_percentage + 0.05)

    def get_speed(self):
        return np.linalg.norm(self.ball_vel)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Draw trail
        if self.trail:
            for i, pos in enumerate(self.trail):
                alpha = int(255 * (i / self.TRAIL_LENGTH))
                color = (*self.COLOR_WALLS, alpha)
                radius = int(self.BALL_RADIUS * 0.5 * (i / self.TRAIL_LENGTH))
                if radius > 0:
                    trail_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(trail_surf, color, (radius, radius), radius)
                    self.screen.blit(trail_surf, (int(pos[0] - radius), int(pos[1] - radius)))

        # Draw sparks
        for spark in self.sparks:
            alpha = max(0, 255 * (spark['life'] / self.SPARK_LIFETIME))
            spark_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(spark_surf, (*self.COLOR_SPARK, alpha), (2, 2), 2)
            self.screen.blit(spark_surf, (int(spark['pos'][0]-2), int(spark['pos'][1]-2)))


        # Draw ball with glow
        speed_ratio = min(1.0, self.get_speed() / self.TARGET_SPEED)
        ball_color = self.COLOR_BALL_SLOW * (1 - speed_ratio) + self.COLOR_BALL_FAST * speed_ratio
        ball_color = tuple(np.clip(ball_color, 0, 255).astype(int))
        
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        
        # Glow effect
        for i in range(4, 0, -1):
            glow_radius = self.BALL_RADIUS + i * 2
            glow_alpha = 40 - i * 8
            glow_color = (*ball_color, glow_alpha)
            
            glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (ball_center[0] - glow_radius, ball_center[1] - glow_radius))

        # Main ball circle
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, ball_color)

    def _render_ui(self):
        # Current Speed
        speed_text = f"Speed: {int(self.get_speed())} u/s"
        speed_surf = self.font_main.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (15, self.HEIGHT - 40))

        # Timer
        time_left = max(0, self.MAX_TIME - self.time_elapsed)
        timer_text = f"Time: {time_left:.1f}"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 15, 10))

        # Target Speed
        target_text = f"Target: {self.TARGET_SPEED} u/s"
        target_surf = self.font_small.render(target_text, True, self.COLOR_TEXT)
        self.screen.blit(target_surf, ((self.WIDTH - target_surf.get_width()) // 2, 10))
        
        # Speed Decay (persistent display)
        decay_text = f"Decay: {self.speed_decay_percentage:.0%}"
        decay_surf = self.font_small.render(decay_text, True, self.COLOR_TEXT)
        self.screen.blit(decay_surf, (15, 10))

        # Impact Text
        if self.last_collision_info:
            info = self.last_collision_info
            alpha = min(255, int(255 * (info['life'] / (self.FPS * 0.5))))
            color = (*self.COLOR_BALL_FAST, alpha)
            impact_surf = self.font_impact.render(info['text'], True, color)
            pos = (info['pos'][0] - impact_surf.get_width() / 2, info['pos'][1] - impact_surf.get_height() / 2)
            self.screen.blit(impact_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "speed": self.get_speed(),
            "time_left": self.MAX_TIME - self.time_elapsed,
            "speed_decay": self.speed_decay_percentage,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Ensure display is initialized for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Bounce Velocity Challenge")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    # --- Main Game Loop ---
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Speed: {info['speed']:.0f}")

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Reason: {'Win' if info['speed'] >= env.TARGET_SPEED else 'Timeout'}")
            print(f"Next attempt decay: {env.speed_decay_percentage:.0%}")
            print("Press 'R' to restart.")
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()