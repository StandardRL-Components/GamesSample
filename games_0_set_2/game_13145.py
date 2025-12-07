import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:14:10.852437
# Source Brief: brief_03145.md
# Brief Index: 3145
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls the vertical speed of three
    oscillating balls to hit targets.

    The goal is to score 100 points by hitting targets before the 60-second timer
    runs out or all three balls are lost.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Controls Ball 0 (top) and Ball 1 (middle).
        - 1 (Up Arrow): Ball 0 moves up.
        - 2 (Down Arrow): Ball 0 moves down.
        - 3 (Left Arrow): Ball 1 moves up.
        - 4 (Right Arrow): Ball 1 moves down.
    - `action[1]`: Controls Ball 2 (bottom).
        - 1 (Spacebar): Ball 2 moves up.
    - `action[2]`: Controls Ball 2 (bottom).
        - 1 (Shift Key): Ball 2 moves down.

    **Observation Space:** A 640x400 RGB image of the game screen.

    **Rewards:**
    - +10 for hitting a target.
    - +0.1 for each frame a ball moves closer to any target.
    - +100 terminal reward for winning (reaching 100 points).
    - -100 terminal reward for losing all balls.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control three oscillating balls to hit targets and score points before time runs out or you lose all your balls."
    )
    user_guide = (
        "Use ↑/↓ to move the top ball, ←/→ for the middle ball, and Space/Shift for the bottom ball. Hit targets to score."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_TARGET = (0, 200, 255)
    COLOR_WALL = (60, 70, 90)
    COLOR_UI_TEXT = (220, 220, 220)
    BALL_COLORS = [(255, 80, 80), (255, 180, 50), (80, 255, 80)] # Red (1 life), Yellow (2 lives), Green (3 lives)

    # Game Parameters
    BALL_RADIUS = 12
    BALL_MAX_VY = 8.0
    BALL_VY_CHANGE = 0.5
    TARGET_SIZE = (40, 40)
    WALL_THICKNESS = 10
    NUM_TARGETS = 2
    WIN_SCORE = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls = []
        self.targets = []
        self.particles = []
        self.previous_distances = {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # --- Initialize Balls ---
        self.balls = []
        for i in range(3):
            self.balls.append({
                "id": i,
                "pos": pygame.Vector2(
                    self.SCREEN_WIDTH / 2,
                    (self.SCREEN_HEIGHT / 4) * (i + 1)
                ),
                "vy": 0,
                "lives": 3,
                "phase": i * (2 * math.pi / 3) # Phase offset for oscillation
            })

        # --- Initialize Targets ---
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._respawn_target()

        # --- Initialize Particles for background ---
        self.particles = []
        for _ in range(100):
            self.particles.append([
                random.randint(0, self.SCREEN_WIDTH),
                random.randint(0, self.SCREEN_HEIGHT),
                random.uniform(0.5, 2.0), # radius
                random.uniform(0.1, 0.5) # drift speed
            ])
            
        # --- Initialize distance tracking for reward ---
        self.previous_distances = {
            ball["id"]: self._get_closest_target_dist(ball["pos"])[0]
            for ball in self.balls
        }

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Actions ---
        self._handle_actions(action)

        # --- 2. Update Game State ---
        reward_from_update = self._update_game_state()
        reward += reward_from_update

        # --- 3. Calculate Rewards ---
        # Event-based rewards are calculated inside _update_game_state
        # Here we calculate continuous and terminal rewards.
        reward += self._calculate_reward()
        
        # --- 4. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif not self.balls:
                reward -= 100 # Lose all balls penalty
            # No specific penalty for time out, just the absence of a win bonus.

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Ball 0 control (Up/Down)
        if movement == 1 and self.balls:
            ball_0 = self._get_ball_by_id(0)
            if ball_0: ball_0["vy"] -= self.BALL_VY_CHANGE
        elif movement == 2 and self.balls:
            ball_0 = self._get_ball_by_id(0)
            if ball_0: ball_0["vy"] += self.BALL_VY_CHANGE

        # Ball 1 control (Left/Right as Up/Down)
        if movement == 3 and self.balls:
            ball_1 = self._get_ball_by_id(1)
            if ball_1: ball_1["vy"] -= self.BALL_VY_CHANGE
        elif movement == 4 and self.balls:
            ball_1 = self._get_ball_by_id(1)
            if ball_1: ball_1["vy"] += self.BALL_VY_CHANGE
        
        # Ball 2 control (Space/Shift)
        ball_2 = self._get_ball_by_id(2)
        if ball_2:
            if space_held:
                ball_2["vy"] -= self.BALL_VY_CHANGE
            if shift_held:
                ball_2["vy"] += self.BALL_VY_CHANGE
                
    def _update_game_state(self):
        time = self.steps / self.FPS
        reward = 0
        
        # Update background particles
        for p in self.particles:
            p[1] += p[3] # Move down
            if p[1] > self.SCREEN_HEIGHT:
                p[1] = 0
                p[0] = random.randint(0, self.SCREEN_WIDTH)

        # Update balls
        for ball in self.balls[:]: # Iterate over a copy
            # Clamp vertical velocity
            ball["vy"] = max(-self.BALL_MAX_VY, min(self.BALL_MAX_VY, ball["vy"]))
            
            # Update position
            ball["pos"].y += ball["vy"]
            amplitude = (self.SCREEN_WIDTH / 2) - self.WALL_THICKNESS - self.BALL_RADIUS - 5
            center_x = self.SCREEN_WIDTH / 2
            ball["pos"].x = center_x + amplitude * math.sin(2 * math.pi * 2 * time + ball["phase"]) # 2Hz oscillation

            # --- Wall Collisions ---
            # Top/Bottom bounce
            if ball["pos"].y - self.BALL_RADIUS < self.WALL_THICKNESS:
                ball["pos"].y = self.WALL_THICKNESS + self.BALL_RADIUS
                ball["vy"] *= -0.9 # Dampen bounce
            if ball["pos"].y + self.BALL_RADIUS > self.SCREEN_HEIGHT - self.WALL_THICKNESS:
                ball["pos"].y = self.SCREEN_HEIGHT - self.WALL_THICKNESS - self.BALL_RADIUS
                ball["vy"] *= -0.9 # Dampen bounce
                
            # Left/Right wall hit (lose life)
            # A bit of a buffer to make the color change visible before it's gone
            if ball["pos"].x - self.BALL_RADIUS < self.WALL_THICKNESS + 5 or \
               ball["pos"].x + self.BALL_RADIUS > self.SCREEN_WIDTH - self.WALL_THICKNESS - 5:
                # This check prevents losing multiple lives in one long contact
                if not ball.get("hit_wall_this_frame", False):
                    ball["lives"] -= 1
                    ball["hit_wall_this_frame"] = True
                    if ball["lives"] <= 0:
                        self.balls.remove(ball)
                        continue # Skip to next ball
            else:
                ball["hit_wall_this_frame"] = False
                
            # --- Target Collisions ---
            ball_rect = pygame.Rect(ball["pos"].x - self.BALL_RADIUS, ball["pos"].y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            for target in self.targets[:]:
                if target.colliderect(ball_rect):
                    self.score += 10
                    reward += 10
                    self.targets.remove(target)
                    self._respawn_target()
                    break # A ball can only hit one target per frame
        return reward

    def _calculate_reward(self):
        reward = 0
        current_distances = {}
        for ball in self.balls:
            # Calculate distance-based reward
            dist, _ = self._get_closest_target_dist(ball["pos"])
            current_distances[ball["id"]] = dist
            
            prev_dist = self.previous_distances.get(ball["id"])
            if prev_dist is not None and dist < prev_dist:
                reward += 0.1

        self.previous_distances = current_distances
        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if not self.balls: # All balls are lost
            return True
        return False

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
        # Render background particles
        for p in self.particles:
            color_val = int(p[2] * 40)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(p[0]), int(p[1])), int(p[2]))

        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.SCREEN_HEIGHT - self.WALL_THICKNESS, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        
        # Render targets
        for target in self.targets:
            pygame.draw.rect(self.screen, self.COLOR_TARGET, target, border_radius=5)
            
        # Render balls
        for ball in self.balls:
            pos = (int(ball["pos"].x), int(ball["pos"].y))
            lives = ball["lives"]
            if 1 <= lives <= 3:
                color = self.BALL_COLORS[lives - 1]
                # Glow effect
                glow_radius = int(self.BALL_RADIUS * 1.8)
                glow_alpha = 80
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
                
                # Main ball
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"Time: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)
        
        # Ball lives indicators
        for ball in self.balls:
            ball_id = ball["id"]
            lives = ball["lives"]
            color = self.BALL_COLORS[lives - 1] if 1 <= lives <= 3 else (50,50,50)
            text = self.font_small.render(f"Ball {ball_id+1}: {'♥' * lives}", True, color)
            self.screen.blit(text, (20, 70 + ball_id * 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": len(self.balls),
            "time_remaining_steps": self.MAX_STEPS - self.steps,
        }

    def _respawn_target(self):
        # Ensure target doesn't spawn on top of another
        while True:
            new_target = pygame.Rect(
                random.randint(self.WALL_THICKNESS + 20, self.SCREEN_WIDTH - self.TARGET_SIZE[0] - self.WALL_THICKNESS - 20),
                random.randint(self.WALL_THICKNESS + 20, self.SCREEN_HEIGHT - self.TARGET_SIZE[1] - self.WALL_THICKNESS - 20),
                self.TARGET_SIZE[0],
                self.TARGET_SIZE[1]
            )
            if not any(t.colliderect(new_target) for t in self.targets):
                self.targets.append(new_target)
                break

    def _get_ball_by_id(self, ball_id):
        for ball in self.balls:
            if ball["id"] == ball_id:
                return ball
        return None

    def _get_closest_target_dist(self, pos):
        if not self.targets:
            return self.SCREEN_WIDTH, None # Return a large distance if no targets
        
        min_dist = float('inf')
        closest_target = None
        for target in self.targets:
            dist = pos.distance_to(pygame.Vector2(target.center))
            if dist < min_dist:
                min_dist = dist
                closest_target = target
        return min_dist, closest_target

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    # If you are running in a headless environment, this block will fail.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Ball Oscillator")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Human Controls ---
            # Default action is "do nothing"
            action = [0, 0, 0] # [movement, space, shift]
            
            keys = pygame.key.get_pressed()
            
            # Ball 0
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            
            # Ball 1
            if keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Ball 2
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0

            # --- Rendering ---
            # The observation is already a rendered frame
            # We just need to convert it back to a Pygame Surface and display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                print("Press 'R' to restart.")
                # Wait for reset
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        break
                    if event.type == pygame.QUIT:
                        running = False
                        break
            
            clock.tick(GameEnv.FPS)

        env.close()
    except pygame.error as e:
        print(f"Could not run graphical test: {e}")
        print("This is expected in a headless environment.")
        # Create a dummy env to ensure it initializes without error
        print("Testing headless initialization...")
        env = GameEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("Headless initialization successful.")