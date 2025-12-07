import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:31:20.243499
# Source Brief: brief_02219.md
# Brief Index: 2219
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Grow a fractal plant to match a target pattern. Use portals to manipulate the growth rate "
        "and replicate the ghostly image before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to plant the seed. "
        "Hold shift and press space to place a growth-slowing portal. Hold shift and use arrow keys to move a selected portal."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds episode

    # Colors
    COLOR_BG = (15, 19, 25)
    COLOR_GRID = (30, 35, 45)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TARGET = (255, 255, 255, 50) # Ghostly white
    COLOR_FRACTAL = (0, 255, 128)
    COLOR_FRACTAL_STUNTED = (255, 80, 80)
    COLOR_PORTAL_SLOW = (50, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Game mechanics
    CURSOR_SPEED = 8
    PORTAL_RADIUS = 60
    PORTAL_EFFECT = 0.5 # Slows growth to 50%
    MAX_PORTALS = 3
    FRACTAL_MAX_GEN = 6
    FRACTAL_BASE_LENGTH = 50
    FRACTAL_LENGTH_DECAY = 0.75
    FRACTAL_ANGLE_SPLIT = 30 # degrees
    WIN_THRESHOLD = 0.85 # 85% similarity to win

    class Branch:
        def __init__(self, start_pos, angle, generation, base_length, angle_split, length_decay):
            self.start_pos = start_pos
            self.angle = angle
            self.generation = generation
            self.is_alive = True
            self.children = []
            self.color = GameEnv.COLOR_FRACTAL

            self.target_length = base_length * (length_decay ** generation)
            self.current_length = 0
            
            rad_angle = math.radians(self.angle)
            self.end_pos = (
                self.start_pos[0] + self.current_length * math.cos(rad_angle),
                self.start_pos[1] + self.current_length * math.sin(rad_angle)
            )

        def grow(self, growth_rate, max_gen, base_length, angle_split, length_decay):
            if not self.is_alive:
                for child in self.children:
                    child.grow(growth_rate, max_gen, base_length, angle_split, length_decay)
                return

            self.current_length += growth_rate
            
            if growth_rate < 1.0:
                self.color = GameEnv.COLOR_FRACTAL_STUNTED
            else:
                self.color = GameEnv.COLOR_FRACTAL

            if self.current_length >= self.target_length:
                self.current_length = self.target_length
                self.is_alive = False
                if self.generation < max_gen:
                    # SFX: Soft chime for new branch
                    self.children.append(GameEnv.Branch(self.end_pos, self.angle - angle_split, self.generation + 1, base_length, angle_split, length_decay))
                    self.children.append(GameEnv.Branch(self.end_pos, self.angle + angle_split, self.generation + 1, base_length, angle_split, length_decay))

            rad_angle = math.radians(self.angle)
            self.end_pos = (
                self.start_pos[0] + self.current_length * math.cos(rad_angle),
                self.start_pos[1] + self.current_length * math.sin(rad_angle)
            )

        def get_segments(self):
            segments = [((int(self.start_pos[0]), int(self.start_pos[1])), (int(self.end_pos[0]), int(self.end_pos[1])), self.color, max(1, 4 - self.generation))]
            for child in self.children:
                segments.extend(child.get_segments())
            return segments

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
        self.font_ui = pygame.font.SysFont("consolas", 20)
        self.font_msg = pygame.font.SysFont("consolas", 30, bold=True)
        
        self.render_mode = render_mode
        self._initialize_state()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.portals = []
        self.selected_portal_idx = -1
        self.fractal_root = None
        self.target_fractal_segments = []
        self.target_fractal_cells = set()
        self.time_left = self.MAX_STEPS
        self.last_similarity = 0
        self.win_message = ""
        
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        
        if options and "level" in options:
            self.level = options["level"]
        else:
            self.level += 1
            
        self._generate_target_fractal()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = False # This env doesn't truncate
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True
            # SFX: Win or Lose sound
            if self.win_message == "PATTERN MATCHED":
                self.score += 100
                reward += 100
            elif self.win_message == "TIME'S UP":
                self.score -= 100
                reward -= 100

        self.steps += 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_code, shift_code = action
        space_held = space_code == 1
        shift_held = shift_code == 1
        
        space_press = space_held and not self.prev_space_held

        # --- Movement ---
        move_vec = np.array([0, 0], dtype=float)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1  # Right
        
        if shift_held and self.selected_portal_idx != -1:
            # Move selected portal
            self.portals[self.selected_portal_idx] += move_vec * self.CURSOR_SPEED
            self.portals[self.selected_portal_idx][0] = np.clip(self.portals[self.selected_portal_idx][0], 0, self.SCREEN_WIDTH)
            self.portals[self.selected_portal_idx][1] = np.clip(self.portals[self.selected_portal_idx][1], 0, self.SCREEN_HEIGHT)
        else:
            # Move cursor
            self.cursor_pos += move_vec * self.CURSOR_SPEED
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Actions ---
        if space_press:
            if shift_held:
                # Place a new portal
                if len(self.portals) < self.MAX_PORTALS:
                    # SFX: Portal placement woosh
                    self.portals.append(self.cursor_pos.copy())
                    self.selected_portal_idx = len(self.portals) - 1
            else:
                # Plant a seed
                if self.fractal_root is None:
                    # SFX: Seed planting pop
                    self.fractal_root = self.Branch(
                        start_pos=tuple(self.cursor_pos),
                        angle=-90,
                        generation=0,
                        base_length=self.FRACTAL_BASE_LENGTH,
                        angle_split=self.FRACTAL_ANGLE_SPLIT,
                        length_decay=self.FRACTAL_LENGTH_DECAY
                    )

        # Update previous state for press detection
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        self.time_left -= 1
        
        if self.fractal_root is not None:
            # Determine growth rate at branch locations
            # This is a simplification; a more complex model would check along the branch
            growth_rate = 1.0
            if self.fractal_root.is_alive:
                pos = self.fractal_root.start_pos
                for portal_pos in self.portals:
                    if np.linalg.norm(np.array(pos) - portal_pos) < self.PORTAL_RADIUS:
                        growth_rate = self.PORTAL_EFFECT
                        break
            
            # Recursively grow the fractal tree
            self.fractal_root.grow(
                growth_rate=growth_rate,
                max_gen=self.FRACTAL_MAX_GEN,
                base_length=self.FRACTAL_BASE_LENGTH,
                angle_split=self.FRACTAL_ANGLE_SPLIT,
                length_decay=self.FRACTAL_LENGTH_DECAY
            )

    def _calculate_similarity(self):
        if not self.fractal_root:
            return 0.0

        grown_segments = self.fractal_root.get_segments()
        if not grown_segments:
            return 0.0

        # Discretize grown fractal onto a grid
        scale = 10
        grown_cells = set()
        for start, end, _, _ in grown_segments:
            dist = math.hypot(end[0] - start[0], end[1] - start[1])
            if dist == 0: continue
            steps = int(dist / (scale/2)) + 1
            for i in range(steps):
                t = i / (steps-1) if steps > 1 else 0
                px = int(start[0] * (1-t) + end[0] * t)
                py = int(start[1] * (1-t) + end[1] * t)
                grown_cells.add((px // scale, py // scale))

        # Jaccard Similarity
        intersection = len(self.target_fractal_cells.intersection(grown_cells))
        union = len(self.target_fractal_cells.union(grown_cells))
        
        return intersection / union if union > 0 else 0

    def _calculate_reward(self):
        similarity = self._calculate_similarity()
        reward = (similarity - self.last_similarity) * 10 # Reward for increasing similarity
        self.last_similarity = similarity
        return reward

    def _check_termination(self):
        if self.time_left <= 0:
            self.win_message = "TIME'S UP"
            return True
        if self.last_similarity >= self.WIN_THRESHOLD:
            self.win_message = "PATTERN MATCHED"
            return True
        return False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "similarity": self.last_similarity,
        }

    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        # Game elements
        self._render_target_fractal()
        self._render_portals()
        self._render_growing_fractal()
        self._render_cursor()
        
        # UI
        self._render_ui()
        if self.game_over:
            self._render_game_over_message()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_target_fractal(self):
        for start, end, _, width in self.target_fractal_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TARGET, start, end)

    def _render_growing_fractal(self):
        if self.fractal_root:
            segments = self.fractal_root.get_segments()
            for start, end, color, width in segments:
                if start == end: continue
                pygame.draw.line(self.screen, color, start, end, width)

    def _render_portals(self):
        for i, pos in enumerate(self.portals):
            px, py = int(pos[0]), int(pos[1])
            is_selected = (i == self.selected_portal_idx)
            
            # Pulsating effect
            pulse = abs(math.sin(self.steps * 0.1))
            
            # Outer ring
            color = self.COLOR_PORTAL_SLOW
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PORTAL_RADIUS, (*color, int(100 + pulse * 100)))
            
            # Inner rotating elements
            for j in range(3):
                angle = self.steps * 0.02 + (j * 2 * math.pi / 3)
                ix = px + int(math.cos(angle) * self.PORTAL_RADIUS * 0.7)
                iy = py + int(math.sin(angle) * self.PORTAL_RADIUS * 0.7)
                pygame.gfxdraw.filled_circle(self.screen, ix, iy, 3, (*color, int(150 + pulse * 105)))
                
            if is_selected:
                pygame.gfxdraw.aacircle(self.screen, px, py, self.PORTAL_RADIUS + 3, (255, 255, 0, int(150 + pulse * 105)))

    def _render_cursor(self):
        px, py = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        size = 8
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py), (px + size, py), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (px, py - size), (px, py + size), 2)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Similarity
        sim_text = f"SIMILARITY: {self.last_similarity*100:.1f}%"
        sim_surf = self.font_ui.render(sim_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(sim_surf, (self.SCREEN_WIDTH // 2 - sim_surf.get_width() // 2, 10))

    def _render_game_over_message(self):
        msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_CURSOR)
        pos = (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2)
        
        # Simple drop shadow for readability
        shadow_surf = self.font_msg.render(self.win_message, True, (0,0,0))
        self.screen.blit(shadow_surf, (pos[0]+2, pos[1]+2))
        self.screen.blit(msg_surf, pos)

    def _generate_target_fractal(self):
        self.target_fractal_segments.clear()
        self.target_fractal_cells.clear()
        
        num_branches = min(3 + self.level // 2, 8)
        base_angle = -90
        
        def _branch(start_pos, angle, generation, length):
            if generation > num_branches:
                return
            
            rad_angle = math.radians(angle)
            end_pos = (
                start_pos[0] + length * math.cos(rad_angle),
                start_pos[1] + length * math.sin(rad_angle)
            )
            
            width = max(1, 4 - generation)
            self.target_fractal_segments.append(((int(start_pos[0]), int(start_pos[1])), (int(end_pos[0]), int(end_pos[1])), self.COLOR_TARGET, width))
            
            new_length = length * self.FRACTAL_LENGTH_DECAY
            angle_split = self.FRACTAL_ANGLE_SPLIT * (0.9**generation)
            
            _branch(end_pos, angle - angle_split, generation + 1, new_length)
            _branch(end_pos, angle + angle_split, generation + 1, new_length)

        start_point = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        _branch(start_point, base_angle, 0, self.FRACTAL_BASE_LENGTH)
        
        # Pre-calculate discretized cells for similarity check
        scale = 10
        for start, end, _, _ in self.target_fractal_segments:
            dist = math.hypot(end[0] - start[0], end[1] - start[1])
            if dist == 0: continue
            steps = int(dist / (scale/2)) + 1
            for i in range(steps):
                t = i / (steps-1) if steps > 1 else 0
                px = int(start[0] * (1-t) + end[0] * t)
                py = int(start[1] * (1-t) + end[1] * t)
                self.target_fractal_cells.add((px // scale, py // scale))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Gardener")
    
    while running:
        movement = 0 # None
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Similarity: {info['similarity']*100:.1f}%")
            pygame.time.wait(3000) # Pause for 3 seconds before reset
            obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    env.close()