
# Generated: 2025-08-27T13:23:50.129297
# Source Brief: brief_00350.md
# Brief Index: 350

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to steer your vehicle. "
        "Stay on the track and pass through green checkpoints to gain time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling racer with a Tron-like aesthetic. "
        "Navigate a procedurally generated neon track, hit checkpoints for time bonuses, "
        "and reach the finish line before time runs out or you crash."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_TRACK = (0, 200, 255)
        self.COLOR_TRACK_GLOW = (0, 100, 150)
        self.COLOR_CAR = (255, 0, 100)
        self.COLOR_CAR_GLOW = (150, 0, 50)
        self.COLOR_CHECKPOINT = (0, 255, 100)
        self.COLOR_FINISH = (255, 215, 0)
        self.COLOR_SPARK = (255, 120, 0)
        self.COLOR_UI = (220, 220, 255)
        self.COLOR_GAMEOVER = (255, 50, 50)

        # Game constants
        self.CAR_X_POS = self.WIDTH // 4
        self.CAR_SPEED = 5.0
        self.SCROLL_SPEED = 4.0
        self.INITIAL_TRACK_WIDTH = 120
        self.TRACK_SEGMENT_LENGTH = 40
        self.NUM_CHECKPOINTS = 5
        self.CHECKPOINT_BONUS_TIME = 300 # steps
        self.INITIAL_TIME = 1800 # 60s @ 30fps
        self.MAX_EPISODE_STEPS = self.INITIAL_TIME + self.NUM_CHECKPOINTS * self.CHECKPOINT_BONUS_TIME + 200
        
        # Initialize state variables
        self.car_y = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.checkpoints_hit = 0
        self.track_centerline = []
        self.checkpoints = []
        self.finish_line = None
        self.track_width = 0
        self.y_variance = 0
        self.particles = []
        self.car_trail = deque(maxlen=20)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.INITIAL_TIME
        self.checkpoints_hit = 0
        self.track_width = self.INITIAL_TRACK_WIDTH
        self.y_variance = 5.0

        self.car_y = self.HEIGHT / 2
        self.particles.clear()
        self.car_trail.clear()

        self._generate_track()

        return self._get_observation(), self._get_info()

    def _generate_track(self):
        self.track_centerline.clear()
        self.checkpoints.clear()

        # Start track off-screen to the right
        current_y = self.HEIGHT / 2
        num_segments = (self.NUM_CHECKPOINTS + 1) * 10
        track_total_length = num_segments * self.TRACK_SEGMENT_LENGTH

        for i in range(num_segments + 20): # Add extra segments for smooth ending
            x = i * self.TRACK_SEGMENT_LENGTH
            y_change = self.np_random.uniform(-self.y_variance, self.y_variance)
            current_y = np.clip(current_y + y_change, self.track_width, self.HEIGHT - self.track_width)
            self.track_centerline.append(pygame.Vector2(x, current_y))

        # Place checkpoints and finish line
        for i in range(1, self.NUM_CHECKPOINTS + 1):
            segment_index = i * 10
            pos = self.track_centerline[segment_index]
            self.checkpoints.append({"pos": pos.copy(), "hit": False, "radius": self.track_width / 2})

        finish_pos = self.track_centerline[num_segments]
        self.finish_line = {"pos": finish_pos.copy(), "hit": False, "width": self.track_width}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        # Update game logic
        self.steps += 1
        self.time_left -= 1
        reward = 0.1  # Survival reward

        # === Player Movement ===
        if movement == 1: # Up
            self.car_y -= self.CAR_SPEED
        elif movement == 2: # Down
            self.car_y += self.CAR_SPEED
        self.car_y = np.clip(self.car_y, 0, self.HEIGHT)
        self.car_trail.append(pygame.Vector2(self.CAR_X_POS, self.car_y))

        # === Scroll Track ===
        for p in self.track_centerline:
            p.x -= self.SCROLL_SPEED
        for cp in self.checkpoints:
            cp["pos"].x -= self.SCROLL_SPEED
        if self.finish_line:
            self.finish_line["pos"].x -= self.SCROLL_SPEED

        # === Collision Detection ===
        y_top, y_bottom = self._get_track_bounds_at_car()
        if not (y_top < self.car_y < y_bottom):
            self.game_over = True
            reward = -100
            self._create_spark_particles(20)
            # sfx: crash_sound
        
        # === Checkpoint & Finish Logic ===
        if not self.game_over:
            # Checkpoints
            for cp in self.checkpoints:
                if not cp["hit"] and cp["pos"].x < self.CAR_X_POS:
                    dist = abs(self.car_y - cp["pos"].y)
                    if dist < cp["radius"]:
                        cp["hit"] = True
                        self.checkpoints_hit += 1
                        self.time_left += self.CHECKPOINT_BONUS_TIME
                        reward += 10
                        self.score += 10
                        self.y_variance += 1.0 # Increase difficulty
                        # sfx: checkpoint_hit
                    else: # Missed checkpoint
                        self.game_over = True
                        reward = -100
                        self._create_spark_particles(20)
                        # sfx: miss_checkpoint_sound
                        break

            # Finish Line
            if self.finish_line and not self.finish_line["hit"] and self.finish_line["pos"].x < self.CAR_X_POS:
                dist = abs(self.car_y - self.finish_line["pos"].y)
                if dist < self.finish_line["width"] / 2:
                    self.finish_line["hit"] = True
                    self.game_over = True
                    reward += 100
                    self.score += 100
                    # sfx: victory_fanfare
                else: # Missed finish
                    self.game_over = True
                    reward = -100
                    self._create_spark_particles(20)
                    # sfx: miss_finish_sound

        # === Particle Update ===
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # === Termination Conditions ===
        terminated = self.game_over or self.time_left <= 0 or self.steps >= self.MAX_EPISODE_STEPS
        if terminated and not self.game_over: # Ran out of time
            self.game_over = True
            reward = -50 # Penalty for timeout
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_track_bounds_at_car(self):
        # Find segment of track the car is on
        p1, p2 = None, None
        for i in range(len(self.track_centerline) - 1):
            if self.track_centerline[i].x <= self.CAR_X_POS < self.track_centerline[i+1].x:
                p1 = self.track_centerline[i]
                p2 = self.track_centerline[i+1]
                break
        
        if p1 is None or p2 is None:
            return -1, self.HEIGHT + 1 # Off track, definite collision

        # Interpolate center y
        t = (self.CAR_X_POS - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
        center_y = p1.y + t * (p2.y - p1.y)

        return center_y - self.track_width / 2, center_y + self.track_width / 2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Prepare track points
        visible_points = [p for p in self.track_centerline if -self.TRACK_SEGMENT_LENGTH < p.x < self.WIDTH + self.TRACK_SEGMENT_LENGTH]
        if len(visible_points) < 2: return
        
        top_wall = [(p.x, p.y - self.track_width / 2) for p in visible_points]
        bottom_wall = [(p.x, p.y + self.track_width / 2) for p in visible_points]

        # Draw track glow
        pygame.draw.lines(self.screen, self.COLOR_TRACK_GLOW, False, top_wall, int(self.track_width * 0.2))
        pygame.draw.lines(self.screen, self.COLOR_TRACK_GLOW, False, bottom_wall, int(self.track_width * 0.2))

        # Draw main track lines
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, top_wall, 5)
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, bottom_wall, 5)

        # Draw checkpoints
        for cp in self.checkpoints:
            if not cp["hit"] and 0 < cp["pos"].x < self.WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, int(cp["pos"].x), int(cp["pos"].y), int(cp["radius"]), (*self.COLOR_CHECKPOINT, 50))
                pygame.gfxdraw.aacircle(self.screen, int(cp["pos"].x), int(cp["pos"].y), int(cp["radius"]), self.COLOR_CHECKPOINT)

        # Draw finish line
        if self.finish_line and not self.finish_line["hit"] and 0 < self.finish_line["pos"].x < self.WIDTH:
            pos = self.finish_line["pos"]
            width = self.finish_line["width"]
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 20
            pygame.draw.rect(self.screen, (*self.COLOR_FINISH, 50), (pos.x - 5, pos.y - width/2 - pulse/2, 10, width + pulse))
            pygame.draw.rect(self.screen, self.COLOR_FINISH, (pos.x - 2, pos.y - width/2, 4, width))
            
        # Draw car trail
        for i, pos in enumerate(self.car_trail):
            alpha = int(255 * (i / len(self.car_trail)))
            if i > 1:
                pygame.draw.line(self.screen, (*self.COLOR_CAR, alpha), self.car_trail[i-1], pos, max(1, int(8 * (i / len(self.car_trail)))))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            pygame.draw.circle(self.screen, (*self.COLOR_SPARK, alpha), p["pos"], int(p["life"] / 2))

        # Draw car
        if not self.game_over:
            p1 = (self.CAR_X_POS + 15, self.car_y)
            p2 = (self.CAR_X_POS - 10, self.car_y - 8)
            p3 = (self.CAR_X_POS - 10, self.car_y + 8)
            # Glow
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), (*self.COLOR_CAR_GLOW, 100))
            # Body
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_CAR)
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_CAR)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {max(0, self.time_left // 30):02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Checkpoints
        cp_text = f"CHECKPOINTS: {self.checkpoints_hit} / {self.NUM_CHECKPOINTS}"
        cp_surf = self.font_small.render(cp_text, True, self.COLOR_UI)
        self.screen.blit(cp_surf, (self.WIDTH // 2 - cp_surf.get_width() // 2, self.HEIGHT - 30))

        # Game Over message
        if self.game_over:
            msg = "FINISH!" if self.finish_line and self.finish_line["hit"] else "CRASHED!"
            color = self.COLOR_FINISH if self.finish_line and self.finish_line["hit"] else self.COLOR_GAMEOVER
            go_surf = self.font_large.render(msg, True, color)
            self.screen.blit(go_surf, (self.WIDTH // 2 - go_surf.get_width() // 2, self.HEIGHT // 2 - go_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "checkpoints_hit": self.checkpoints_hit,
        }
        
    def _create_spark_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(self.CAR_X_POS, self.car_y),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "life": life,
                "max_life": life,
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Line Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # In a real game, you might wait for the reset key here
            # For this example, we'll just let it sit on the 'game over' screen
            
        clock.tick(30) # Run at 30 FPS

    env.close()