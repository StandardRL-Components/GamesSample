import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:30:27.678779
# Source Brief: brief_00447.md
# Brief Index: 447
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a flock of birds through
    procedurally generated obstacle courses. The goal is to maintain flock
    cohesion and complete multiple levels within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a flock of birds through procedurally generated obstacle courses. "
        "Maintain flock cohesion and navigate all levels before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to steer the selected bird. Press space to cycle to the next bird, "
        "and shift to cycle to the previous one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.NUM_BIRDS = 5
        self.COHESION_RADIUS = 100
        self.OBSTACLE_BASE_COUNT = 6
        self.LEVEL_COUNT = 3
        self.MAX_GAME_SECONDS = 60.0
        self.FPS = 30
        self.MAX_STEPS = int(self.MAX_GAME_SECONDS * self.FPS)

        # --- Colors ---
        self.COLOR_BG = (26, 33, 41)
        self.COLOR_OBSTACLE = (80, 90, 100)
        self.COLOR_COHESION = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PROGRESS_BAR = (70, 180, 120)
        self.COLOR_PROGRESS_BAR_BG = (40, 50, 60)
        self.COLOR_SELECTED_GLOW = (255, 255, 255)

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
        try:
            self.font = pygame.font.SysFont("dejavusansmono", 20, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont("monospace", 20, bold=True)

        # --- Internal State (initialized in reset) ---
        self.birds = []
        self.obstacles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level = 1
        self.total_game_time = 0.0
        self.selected_bird_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.win_condition_met = False
        self.flock_center = np.zeros(2)

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is a helper, not part of the API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.level = 1
        self.total_game_time = 0.0
        self.selected_bird_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.win_condition_met = False

        self.birds = []
        min_speed, max_speed = 40, 90
        for i in range(self.NUM_BIRDS):
            speed_ratio = (self.NUM_BIRDS > 1) and (i / (self.NUM_BIRDS - 1)) or 0.5
            speed = min_speed + (max_speed - min_speed) * speed_ratio
            color = self._get_speed_color(speed_ratio)
            start_pos = np.array([
                80.0 + self.np_random.uniform(-30, 30),
                self.SCREEN_HEIGHT / 2 + self.np_random.uniform(-30, 30)
            ])
            self.birds.append({
                "pos": start_pos,
                "vel": np.array([1.0, 0.0]),
                "speed": speed,
                "color": color,
                "trail": []
            })

        self.flock_center = self._calculate_flock_center()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self._handle_input(action)
        self._update_physics()
        reward, terminated = self._calculate_reward_and_termination()

        self.score += reward
        self.game_over = terminated
        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if space_held and not self.prev_space_held:
            self.selected_bird_idx = (self.selected_bird_idx + 1) % self.NUM_BIRDS
            # sfx: select_next_bird.play()
        if shift_held and not self.prev_shift_held:
            self.selected_bird_idx = (self.selected_bird_idx - 1 + self.NUM_BIRDS) % self.NUM_BIRDS
            # sfx: select_prev_bird.play()
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        target_dir = None
        if movement == 1: target_dir = np.array([0.0, -1.0]) # Up
        elif movement == 2: target_dir = np.array([0.0, 1.0]) # Down
        elif movement == 3: target_dir = np.array([-1.0, 0.0]) # Left
        elif movement == 4: target_dir = np.array([1.0, 0.0]) # Right

        if target_dir is not None:
            selected_bird = self.birds[self.selected_bird_idx]
            current_dir = selected_bird['vel']
            # Smoothly interpolate towards the target direction for better game feel
            lerp_factor = 0.25
            new_dir = current_dir * (1 - lerp_factor) + target_dir * lerp_factor
            selected_bird['vel'] = new_dir / (np.linalg.norm(new_dir) + 1e-6)

    def _update_physics(self):
        dt = 1.0 / self.FPS
        self.total_game_time += dt
        self.flock_center = self._calculate_flock_center()

        for i, bird in enumerate(self.birds):
            vec_to_center = self.flock_center - bird['pos']
            dist_to_center = np.linalg.norm(vec_to_center)
            
            # Only apply cohesion if the bird is trying to stray
            if dist_to_center > 10:
                cohesion_force = vec_to_center / (dist_to_center + 1e-6)
                cohesion_weight = 0.05 if i == self.selected_bird_idx else 0.15
                
                current_dir = bird['vel']
                final_dir = current_dir * (1 - cohesion_weight) + cohesion_force * cohesion_weight
                bird['vel'] = final_dir / (np.linalg.norm(final_dir) + 1e-6)

            bird['pos'] += bird['vel'] * bird['speed'] * dt
            
            # Screen wrap logic
            if bird['pos'][0] < 0: bird['pos'][0] = self.SCREEN_WIDTH
            if bird['pos'][0] > self.SCREEN_WIDTH: bird['pos'][0] = 0
            if bird['pos'][1] < 0: bird['pos'][1] = self.SCREEN_HEIGHT
            if bird['pos'][1] > self.SCREEN_HEIGHT: bird['pos'][1] = 0

            bird['trail'].append(np.copy(bird['pos']))
            if len(bird['trail']) > 15:
                bird['trail'].pop(0)

    def _calculate_reward_and_termination(self):
        terminated = False
        reward = 0.0

        for bird in self.birds:
            if np.linalg.norm(bird['pos'] - self.flock_center) > self.COHESION_RADIUS:
                # sfx: fail_cohesion.play()
                return -100.0, True
            
            bird_rect = pygame.Rect(bird['pos'][0] - 4, bird['pos'][1] - 4, 8, 8)
            if bird_rect.collidelist(self.obstacles) != -1:
                # sfx: fail_crash.play()
                return -100.0, True

        if all(b['pos'][0] > self.SCREEN_WIDTH - 40 for b in self.birds):
            self.level += 1
            reward += 10.0
            # sfx: level_complete.play()

            if self.level > self.LEVEL_COUNT:
                self.win_condition_met = True
                terminated = True
                time_bonus = max(0, self.MAX_GAME_SECONDS - self.total_game_time)
                reward += 50.0 + time_bonus
                # sfx: game_win.play()
            else:
                self._generate_level()
                for bird in self.birds:
                    bird['pos'][0] = 40.0
                    bird['trail'].clear()

        if self.total_game_time >= self.MAX_GAME_SECONDS:
            if not self.win_condition_met:
                reward = -100.0
            terminated = True
            # sfx: fail_timeout.play()

        if not terminated:
            reward += 0.1

        return reward, terminated

    def _get_observation(self):
        camera_offset = self.flock_center - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        self.screen.fill(self.COLOR_BG)

        self._render_cohesion_circle(camera_offset)
        self._render_obstacles(camera_offset)
        self._render_birds(camera_offset)
        self._render_ui()

        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_birds(self, camera_offset):
        for i, bird in enumerate(self.birds):
            pos = bird['pos'] - camera_offset
            
            if len(bird['trail']) > 1:
                trail_points = [(p - camera_offset).astype(int) for p in bird['trail']]
                pygame.draw.aalines(self.screen, bird['color'], False, trail_points, 1)

            size = 12
            v_norm = bird['vel']
            p1 = pos + v_norm * size * 0.7
            p2 = pos - v_norm * size * 0.3 + np.array([-v_norm[1], v_norm[0]]) * size * 0.4
            p3 = pos - v_norm * size * 0.3 - np.array([-v_norm[1], v_norm[0]]) * size * 0.4
            
            points = [p1.astype(int), p2.astype(int), p3.astype(int)]
            
            if i == self.selected_bird_idx:
                glow_size = size * 1.5
                glow_p1 = pos + v_norm * glow_size * 0.7
                glow_p2 = pos - v_norm * glow_size * 0.3 + np.array([-v_norm[1], v_norm[0]]) * glow_size * 0.4
                glow_p3 = pos - v_norm * glow_size * 0.3 - np.array([-v_norm[1], v_norm[0]]) * glow_size * 0.4
                glow_points = [glow_p1.astype(int), glow_p2.astype(int), glow_p3.astype(int)]
                pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_SELECTED_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_SELECTED_GLOW)

            pygame.gfxdraw.aapolygon(self.screen, points, bird['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, bird['color'])

    def _render_obstacles(self, camera_offset):
        for obs in self.obstacles:
            render_rect = obs.move(-camera_offset[0], -camera_offset[1])
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, render_rect)
            pygame.draw.rect(self.screen, tuple(min(255, c*1.2) for c in self.COLOR_OBSTACLE[:3]), render_rect, 2)


    def _render_cohesion_circle(self, camera_offset):
        center_pos = (self.flock_center - camera_offset).astype(int)
        self._draw_dashed_circle(self.screen, self.COLOR_COHESION, center_pos, self.COHESION_RADIUS, dash_length=12)

    def _render_ui(self):
        level_text = self.font.render(f"Level: {self.level}/{self.LEVEL_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        time_left = max(0, self.MAX_GAME_SECONDS - self.total_game_time)
        timer_text = self.font.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        progress = np.clip(self.flock_center[0] / (self.SCREEN_WIDTH - 40), 0, 1)
        bar_w, bar_h = self.SCREEN_WIDTH - 20, 5
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (10, self.SCREEN_HEIGHT - bar_h - 10, bar_w, bar_h), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, (10, self.SCREEN_HEIGHT - bar_h - 10, bar_w * progress, bar_h), border_radius=2)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        message = "VICTORY" if self.win_condition_met else "GAME OVER"
        try:
            end_font = pygame.font.SysFont("dejavusansmono", 60, bold=True)
        except pygame.error:
            end_font = pygame.font.SysFont("monospace", 60, bold=True)
        end_text = end_font.render(message, True, (255, 255, 255))
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_seconds": self.total_game_time,
            "selected_bird": self.selected_bird_idx
        }

    def _generate_level(self):
        self.obstacles.clear()
        obstacle_count = self.OBSTACLE_BASE_COUNT + int((self.level - 1) * self.OBSTACLE_BASE_COUNT * 0.1)
        
        for _ in range(obstacle_count):
            x = self.np_random.uniform(200, self.SCREEN_WIDTH - 150)
            y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            w = self.np_random.uniform(20, 80)
            h = self.np_random.uniform(20, 80)
            self.obstacles.append(pygame.Rect(int(x), int(y), int(w), int(h)))

    def _calculate_flock_center(self):
        if not self.birds:
            return np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        return np.mean([b['pos'] for b in self.birds], axis=0)

    def _get_speed_color(self, ratio):
        blue = (50, 150, 255)
        red = (255, 80, 80)
        r = int(blue[0] * (1 - ratio) + red[0] * ratio)
        g = int(blue[1] * (1 - ratio) + red[1] * ratio)
        b = int(blue[2] * (1 - ratio) + red[2] * ratio)
        return (r, g, b)

    def _draw_dashed_circle(self, surf, color, center, radius, width=1, dash_length=10):
        circumference = 2 * math.pi * radius
        if circumference == 0: return
        num_dashes = int(circumference / (dash_length * 2))
        if num_dashes < 2: return
        angle_step = 360 / (num_dashes * 2)

        for i in range(0, num_dashes * 2, 2):
            start_angle = math.radians(i * angle_step)
            end_angle = math.radians((i + 1) * angle_step)
            arc_rect = (center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            try:
                pygame.draw.arc(surf, color, arc_rect, start_angle, end_angle, width)
            except TypeError: # Some pygame versions don't like width on aacolor
                pygame.draw.arc(surf, color, arc_rect, start_angle, end_angle)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It is not used by the evaluation system, which interacts with the GameEnv class.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # or 'windows', 'mac', etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Flock Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    running = True
    
    key_to_action = {
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2,
        pygame.K_LEFT: 3, pygame.K_a: 3,
        pygame.K_RIGHT: 4, pygame.K_d: 4,
    }

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("Press any key to restart after game over.")
    print("----------------\n")

    while running:
        action = [0, 0, 0] # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        movement_set = False
        for key, move_action in key_to_action.items():
            if keys[key]:
                action[0] = move_action
                movement_set = True
                break
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if terminated or truncated:
            if any(keys):
                obs, info = env.reset()
                terminated = False
                truncated = False
        else:
            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()