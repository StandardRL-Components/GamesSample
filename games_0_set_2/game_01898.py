
# Generated: 2025-08-27T18:38:15.430112
# Source Brief: brief_01898.md
# Brief Index: 1898

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to turn. The car accelerates automatically. "
        "Avoid obstacles and reach the finish line as fast as you can."
    )

    game_description = (
        "A fast-paced, top-down arcade racer on a procedurally generated track. "
        "Dodge obstacles, hit checkpoints, and race against the clock to get the best time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TRACK_LENGTH = 8000  # Total length of the race track in pixels
        self.TRACK_WIDTH = 150

        # Game constants
        self.CAR_SPEED = 8.0
        self.TURN_RATE = math.radians(4.0)
        self.MAX_TIME = 30  # seconds
        self.MAX_COLLISIONS = 5
        self.MAX_STEPS = 1000
        self.FPS = 30

        # Colors
        self.COLOR_BG = (18, 18, 18)
        self.COLOR_TRACK = (40, 40, 40)
        self.COLOR_LINES = (180, 180, 180)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 60)
        self.COLOR_OBSTACLE = (50, 150, 255)
        self.COLOR_CHECKPOINT = (50, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_FINISH_1 = (255, 255, 255)
        self.COLOR_FINISH_2 = (30, 30, 30)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.car_pos = None
        self.car_heading = None
        self.world_offset_y = None
        self.racing_line = None
        self.obstacles = None
        self.checkpoints = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_elapsed = None
        self.collision_count = None
        self.collision_flash_timer = None
        self.current_track_segment_index = None
        self.rng = None

        self.reset()
        
        # Self-validation
        # self.validate_implementation()

    def _generate_track(self):
        """Generates a procedural track."""
        self.racing_line = []
        num_segments = 100
        segment_length = self.TRACK_LENGTH / num_segments
        
        # Track generation parameters
        amplitude = self.rng.uniform(150, 250)
        frequency = self.rng.uniform(0.05, 0.1)
        phase = self.rng.uniform(0, 2 * math.pi)

        for i in range(num_segments + 1):
            x = self.SCREEN_WIDTH / 2 + amplitude * math.sin(frequency * i + phase)
            y = -i * segment_length
            self.racing_line.append(pygame.Vector2(x, y))

        # Generate obstacles
        self.obstacles = []
        obstacle_density = 0.15
        for i in range(1, num_segments):
            if self.rng.random() < obstacle_density + (self.steps / 100) * 0.05:
                point = self.racing_line[i]
                offset = self.rng.uniform(-self.TRACK_WIDTH * 0.7, self.TRACK_WIDTH * 0.7)
                side_vec = (self.racing_line[i+1] - self.racing_line[i-1]).normalize().rotate(90)
                pos = point + side_vec * offset
                radius = self.rng.integers(10, 15)
                self.obstacles.append({'pos': pos, 'radius': radius})
        
        # Generate checkpoints
        self.checkpoints = []
        num_checkpoints = 5
        for i in range(1, num_checkpoints + 1):
            index = int(i * (num_segments / (num_checkpoints + 1)))
            pos = self.racing_line[index]
            self.checkpoints.append({'pos': pos, 'radius': 20, 'active': True})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.car_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.85)
        self.car_heading = math.radians(-90)  # Pointing up
        self.world_offset_y = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.collision_count = 0
        self.collision_flash_timer = 0
        self.current_track_segment_index = 0

        self._generate_track()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        self.time_elapsed += 1.0 / self.FPS
        reward = 0

        # Unpack action
        movement = action[0]
        
        # --- Update Game Logic ---

        # 1. Handle player input
        if movement == 3:  # Left
            self.car_heading -= self.TURN_RATE
        elif movement == 4:  # Right
            self.car_heading += self.TURN_RATE
        
        # Normalize heading
        self.car_heading = (self.car_heading + math.pi) % (2 * math.pi) - math.pi

        # 2. Update car/world position
        move_vec = pygame.Vector2(math.cos(self.car_heading), math.sin(self.car_heading)) * self.CAR_SPEED
        self.world_offset_y -= move_vec.y # World scrolls opposite to car's forward motion
        
        # Lateral movement relative to track center
        # The car doesn't move on screen, the world moves around it.
        # This simulates the car moving left/right in world space.
        for p in self.racing_line: p.x -= move_vec.x
        for o in self.obstacles: o['pos'].x -= move_vec.x
        for c in self.checkpoints: c['pos'].x -= move_vec.x

        # 3. Collision Detection and Rewards
        car_hitbox_radius = 12
        reward += 0.1  # Survival reward

        # Obstacle collisions
        collided_this_frame = False
        for obs in self.obstacles:
            dist = self.car_pos.distance_to(self._world_to_screen(obs['pos']))
            if dist < car_hitbox_radius + obs['radius']:
                if not collided_this_frame:
                    self.collision_count += 1
                    self.collision_flash_timer = 5 # frames
                    # sfx: collision_sound
                    collided_this_frame = True
                # No per-obstacle reward penalty, terminal penalty is enough
                
        # Checkpoint collisions
        for cp in self.checkpoints:
            if cp['active']:
                dist = self.car_pos.distance_to(self._world_to_screen(cp['pos']))
                if dist < car_hitbox_radius + cp['radius']:
                    cp['active'] = False
                    self.score += 1
                    reward += 10.0 # Significant reward for checkpoint
                    # sfx: checkpoint_sound
        
        # Deviation from racing line penalty
        self._update_track_segment()
        if self.current_track_segment_index < len(self.racing_line) - 1:
            p1 = self.racing_line[self.current_track_segment_index]
            p2 = self.racing_line[self.current_track_segment_index + 1]
            track_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            angle_diff = abs(self.car_heading - track_angle)
            if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
            
            if angle_diff > math.radians(15):
                reward -= 0.2
        
        # --- Check Termination Conditions ---
        terminated = False
        finish_y = self.racing_line[-1].y
        
        if self.collision_count >= self.MAX_COLLISIONS:
            reward = -100.0
            terminated = True
        elif self.time_elapsed > self.MAX_TIME:
            reward = -50.0 # Penalty for running out of time
            terminated = True
        elif -self.world_offset_y > self.TRACK_LENGTH: # Reached finish line
            time_bonus = max(0, self.MAX_TIME - self.time_elapsed)
            reward = 100.0 * (time_bonus / self.MAX_TIME)
            self.score += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_track_segment(self):
        """Finds which segment of the racing line the car is currently on."""
        car_world_y = -self.world_offset_y
        for i in range(self.current_track_segment_index, len(self.racing_line) - 1):
            if self.racing_line[i].y > car_world_y > self.racing_line[i+1].y:
                self.current_track_segment_index = i
                return

    def _world_to_screen(self, world_pos):
        """Converts world coordinates to screen coordinates."""
        return pygame.Vector2(world_pos.x, world_pos.y + self.world_offset_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track and boundaries
        for i in range(len(self.racing_line) - 1):
            p1 = self._world_to_screen(self.racing_line[i])
            p2 = self._world_to_screen(self.racing_line[i+1])
            
            # Only draw visible segments
            if max(p1.y, p2.y) > 0 and min(p1.y, p2.y) < self.SCREEN_HEIGHT:
                # Track base
                pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, width=self.TRACK_WIDTH * 2)
                
                # Track boundaries
                direction = (p2 - p1).normalize()
                perp = pygame.Vector2(-direction.y, direction.x)
                
                # Left boundary
                l1 = p1 + perp * self.TRACK_WIDTH
                l2 = p2 + perp * self.TRACK_WIDTH
                pygame.draw.line(self.screen, self.COLOR_LINES, l1, l2, width=5)

                # Right boundary
                r1 = p1 - perp * self.TRACK_WIDTH
                r2 = p2 - perp * self.TRACK_WIDTH
                pygame.draw.line(self.screen, self.COLOR_LINES, r1, r2, width=5)

        # Draw finish line
        finish_pos_y = self._world_to_screen(self.racing_line[-1]).y
        if 0 < finish_pos_y < self.SCREEN_HEIGHT:
            for i in range(0, self.SCREEN_WIDTH, 20):
                color = self.COLOR_FINISH_1 if (i // 20) % 2 == 0 else self.COLOR_FINISH_2
                pygame.draw.rect(self.screen, color, (i, finish_pos_y, 20, 10))

        # Draw checkpoints
        for cp in self.checkpoints:
            if cp['active']:
                pos = self._world_to_screen(cp['pos'])
                if -20 < pos.x < self.SCREEN_WIDTH + 20 and -20 < pos.y < self.SCREEN_HEIGHT + 20:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), cp['radius'], self.COLOR_CHECKPOINT)
                    pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), cp['radius'], self.COLOR_CHECKPOINT)

        # Draw obstacles
        for obs in self.obstacles:
            pos = self._world_to_screen(obs['pos'])
            if -20 < pos.x < self.SCREEN_WIDTH + 20 and -20 < pos.y < self.SCREEN_HEIGHT + 20:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), obs['radius'], self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), obs['radius'], self.COLOR_OBSTACLE)

        # Draw player car
        car_surf = pygame.Surface((30, 16), pygame.SRCALPHA)
        car_rect = pygame.Rect(0, 0, 30, 16)
        pygame.draw.rect(car_surf, self.COLOR_PLAYER, car_rect, border_radius=4)
        
        rotated_car = pygame.transform.rotate(car_surf, -math.degrees(self.car_heading))
        rotated_rect = rotated_car.get_rect(center=self.car_pos)
        
        # Player glow effect
        glow_surf = pygame.Surface((rotated_rect.width + 20, rotated_rect.height + 20), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_surf.get_rect(center=rotated_rect.center))

        self.screen.blit(rotated_car, rotated_rect)

        # Draw collision flash
        if self.collision_flash_timer > 0:
            flash_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.collision_flash_timer / 5.0))
            flash_surf.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surf, (0, 0))
            self.collision_flash_timer -= 1

    def _render_ui(self):
        # Time display
        time_text = f"TIME: {self.time_elapsed:.2f}s"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Collision display
        collision_text = f"HITS: {self.collision_count}/{self.MAX_COLLISIONS}"
        collision_surf = self.font_small.render(collision_text, True, self.COLOR_TEXT)
        self.screen.blit(collision_surf, (self.SCREEN_WIDTH - collision_surf.get_width() - 10, 10))

        # Game over text
        if self.game_over:
            if self.collision_count >= self.MAX_COLLISIONS:
                msg = "CRASHED"
            elif self.time_elapsed > self.MAX_TIME:
                msg = "TIME UP"
            elif -self.world_offset_y > self.TRACK_LENGTH:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            
            end_text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "collisions": self.collision_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- For interactive play ---
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The other actions are unused in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

    print(f"Game Over. Final Info: {info}")
    env.close()