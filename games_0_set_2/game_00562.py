
# Generated: 2025-08-27T14:02:10.726911
# Source Brief: brief_00562.md
# Brief Index: 562

        
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


class SledDrawEnv(gym.Env):
    """
    SledDrawEnv is a physics-based puzzle game where the player draws a track for a sled.
    The goal is to guide the sled from a start point to a finish line within a time limit,
    balancing the creation of a fast track with the need to prevent the sled from crashing.
    Each action adds a new segment to the track, and the sled's movement is then simulated.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to draw the track. ↑/↓ for diagonals, ←/→ for horizontal. "
        "Hold Shift for vertical up, Space for vertical down."
    )

    game_description = (
        "Draw a track for a sled to reach the finish line. Balance speed and safety. "
        "Finish quickly for a bonus, but don't crash!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 300
    SEGMENT_LENGTH = 25.0

    # Colors
    COLOR_BG = (240, 245, 250)
    COLOR_TRACK = (30, 30, 40)
    COLOR_SLED = (230, 50, 50)
    COLOR_START = (60, 200, 60)
    COLOR_FINISH = (230, 50, 50)
    COLOR_UI_TEXT = (50, 50, 50)
    COLOR_PARTICLE = (180, 180, 190)

    # Physics
    GRAVITY_ACCEL = 0.04
    FRICTION = 0.995
    PHYSICS_SUBSTEPS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_game_over = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.crash = False

        self.start_pos = pygame.Vector2(0, 0)
        self.finish_x = 0

        self.track_points = []
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_velocity = 0.0
        self.current_segment_index = 0
        
        self.particles = []
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.crash = False

        self.start_pos = pygame.Vector2(50, 200)
        self.finish_x = self.SCREEN_WIDTH - 50

        self.track_points = [self.start_pos]
        self.sled_pos = pygame.Vector2(self.start_pos)
        self.sled_velocity = 0.0
        self.current_segment_index = 0
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # 1. Unpack action and draw new track segment
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        last_point = self.track_points[-1]
        new_point = pygame.Vector2(last_point)

        # Action Mapping: Shift > Space > Movement
        if shift_held: # Vertical Up
            new_point.y -= self.SEGMENT_LENGTH
        elif space_held: # Vertical Down
            new_point.y += self.SEGMENT_LENGTH
        else:
            if movement == 1: # Up-right diagonal
                angle = -math.pi / 4
                new_point.x += self.SEGMENT_LENGTH * math.cos(angle)
                new_point.y += self.SEGMENT_LENGTH * math.sin(angle)
            elif movement == 2: # Down-right diagonal
                angle = math.pi / 4
                new_point.x += self.SEGMENT_LENGTH * math.cos(angle)
                new_point.y += self.SEGMENT_LENGTH * math.sin(angle)
            elif movement == 3: # Horizontal left
                new_point.x -= self.SEGMENT_LENGTH
            else:  # 0 (none) or 4 (right) -> Horizontal right
                new_point.x += self.SEGMENT_LENGTH
        
        self.track_points.append(new_point)

        # 2. Run internal physics simulation
        for _ in range(self.PHYSICS_SUBSTEPS):
            self._update_physics()
        
        # 3. Update game state and calculate reward
        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_physics(self):
        # Sled can't move if the track isn't long enough
        if self.current_segment_index >= len(self.track_points) - 1:
            self.sled_velocity = 0
            return

        p1 = self.track_points[self.current_segment_index]
        p2 = self.track_points[self.current_segment_index + 1]
        
        segment_vec = p2 - p1
        if segment_vec.length_squared() == 0:
            return

        segment_dir = segment_vec.normalize()
        segment_angle = math.atan2(segment_dir.y, segment_dir.x)

        # Apply gravity based on slope
        gravity_effect = self.GRAVITY_ACCEL * math.sin(segment_angle)
        self.sled_velocity += gravity_effect
        
        # Apply friction
        self.sled_velocity *= self.FRICTION

        # Move sled along the track
        dist_to_move = self.sled_velocity
        
        remaining_dist_on_segment = (p2 - self.sled_pos).length()

        if dist_to_move >= remaining_dist_on_segment and remaining_dist_on_segment > 0:
            # Move to the start of the next segment
            self.sled_pos = pygame.Vector2(p2)
            self.current_segment_index += 1
        elif dist_to_move > 0:
            self.sled_pos += segment_dir * dist_to_move
        
        # Add particles for visual effect
        # sound: sled scraping on ice
        if self.np_random.random() < 0.7:
             self.particles.append(
                 {'pos': pygame.Vector2(self.sled_pos), 'life': 20, 'size': self.np_random.integers(2, 5)}
             )

    def _calculate_reward(self):
        reward = 0.0
        
        # Win condition
        if not self.win and self.sled_pos.x >= self.finish_x:
            self.win = True
            reward += 50.0
            if self.steps < 150:
                reward += 100.0 # Fast completion bonus
            return reward

        # Crash condition
        if not (0 < self.sled_pos.x < self.SCREEN_WIDTH and 0 < self.sled_pos.y < self.SCREEN_HEIGHT):
            self.crash = True
            return -10.0

        # Survival reward
        reward += 0.1
        
        return reward

    def _check_termination(self):
        if self.win:
            return True
        if self.crash:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw start and finish lines
        pygame.draw.circle(self.screen, self.COLOR_START, (int(self.start_pos.x), int(self.start_pos.y)), 10)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.SCREEN_HEIGHT), 5)
        
        # Draw particles
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                alpha = max(0, 255 * (p['life'] / 20.0))
                color = (*self.COLOR_PARTICLE, int(alpha))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color
                )

        # Draw track
        if len(self.track_points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, [(int(p.x), int(p.y)) for p in self.track_points])
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, [(int(p.x), int(p.y)) for p in self.track_points], width=2)
            
        # Draw sled
        sled_size = 8
        sled_rect = pygame.Rect(
            int(self.sled_pos.x - sled_size / 2), 
            int(self.sled_pos.y - sled_size / 2),
            sled_size, sled_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect)
        pygame.gfxdraw.filled_circle(self.screen, int(self.sled_pos.x), int(self.sled_pos.y), sled_size, (*self.COLOR_SLED, 50))

    def _render_ui(self):
        score_surf = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        time_left = self.MAX_STEPS - self.steps
        time_surf = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        if self.game_over:
            msg = ""
            if self.win: msg = "FINISH!"; # sound: cheering crowd
            elif self.crash: msg = "CRASHED!"; # sound: crash sfx
            else: msg = "TIME UP!"; # sound: buzzer
            
            over_surf = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "crash": self.crash,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

# Renaming the class to match the requested `GameEnv` name for the final output.
class GameEnv(SledDrawEnv):
    def __init__(self, render_mode="rgb_array"):
        super().__init__(render_mode)
        # self.validate_implementation() # Uncomment for development

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game and play it with keyboard controls.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sled Draw")
    
    running = True
    game_terminated = False
    
    while running:
        action_taken = False
        action = [0, 0, 0] # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken = True
        
        if action_taken and not game_terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            action = [movement, space_held, shift_held]
            
            obs, reward, game_terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {game_terminated}")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        if game_terminated:
            pygame.time.wait(2000)
            obs, info = env.reset()
            game_terminated = False

        env.clock.tick(30)
        
    env.close()