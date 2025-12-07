import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:21:53.806782
# Source Brief: brief_01649.md
# Brief Index: 1649
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where an agent navigates a shifting dream maze.
    The agent must match word pairs to morph the maze's geometry, creating
    new paths to reach the exit. The environment emphasizes visual quality
    and a fluid gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting dream maze by matching word pairs to alter the level's geometry and create new paths to the exit."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to cycle through words and shift to select a word for matching."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_HEIGHT = 340 # Leave space for UI at the bottom
    MAX_STEPS = 1000
    PLAYER_SPEED = 4
    
    # --- COLORS ---
    COLOR_BG_START = (15, 10, 40)      # Deep Purple
    COLOR_BG_END = (40, 20, 80)        # Indigo
    COLOR_WALL = (60, 80, 160)         # Dark Blue
    COLOR_WALL_MORPHABLE = (100, 120, 200) # Lighter blue for special walls
    COLOR_GROUND = (25, 15, 50)        # Darker than BG
    COLOR_PLAYER = (100, 220, 255)     # Light Blue
    COLOR_PLAYER_GLOW = (50, 150, 255, 100)
    COLOR_EXIT = (255, 180, 80)        # Bright Orange
    COLOR_EXIT_GLOW = (255, 120, 50, 120)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_HIGHLIGHT = (255, 255, 100)
    COLOR_TEXT_SELECTED = (100, 255, 150)
    
    # --- WORD PAIRS ---
    WORD_PAIRS_MASTER = [
        ("DREAM", "SCAPE"), ("MIND", "MAZE"), ("VOID", "ECHO"), ("PATH", "WAY"),
        ("OPEN", "GATE"), ("SILENT", "KEY"), ("LOST", "FOUND"), ("NIGHT", "FALL"),
        ("STAR", "LIGHT"), ("DEEP", "SLEEP"), ("RIFT", "SHIFT"), ("WALL", "FADE")
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_words = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 36)
        
        # --- PRE-RENDERED SURFACES ---
        self.bg_surface = self._create_gradient_background()

        # --- STATE VARIABLES ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 0
        self.total_score = 0
        self.player_pos = np.array([0.0, 0.0])
        self.player_visual_pos = np.array([0.0, 0.0])
        self.maze = None
        self.maze_width = 0
        self.maze_height = 0
        self.cell_size = 0
        self.start_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.words = []
        self.word_map = {}
        self.portal_activations_left = 0
        self.word_cursor_idx = 0
        self.selected_indices = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.dist_to_exit = 0
        self.message = ""
        self.message_timer = 0
        self.ripple_effects = []
        
        # self.reset() is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'level' in options:
            self.level = options['level']
        else:
            self.level = 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.message = ""
        self.message_timer = 0
        self.ripple_effects = []

        self._generate_level()

        self.dist_to_exit = self._calculate_distance_to_exit()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # 1. Handle player movement
        self._move_player(movement)
        
        # 2. Handle word selection logic
        if space_press:
            self.word_cursor_idx = (self.word_cursor_idx + 1) % len(self.words)
            # SFX: UI_Bleep

        if shift_press:
            if self.word_cursor_idx not in self.selected_indices:
                self.selected_indices.append(self.word_cursor_idx)
                # SFX: UI_Select
                if len(self.selected_indices) == 2:
                    match_reward = self._attempt_word_match()
                    reward += match_reward

        # --- Update Game State ---
        self._update_effects()

        # --- Calculate Reward ---
        new_dist = self._calculate_distance_to_exit()
        if new_dist < self.dist_to_exit:
            reward += 0.1  # Moved closer
        elif new_dist > self.dist_to_exit:
            reward -= 0.1  # Moved further
        self.dist_to_exit = new_dist
        
        self.score += reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.player_pos[0] == self.exit_pos[0] and self.player_pos[1] == self.exit_pos[1]:
                self.score += 100
                self.total_score += self.score
                self.level += 1
                self._set_message(f"LEVEL {self.level-1} COMPLETE!", 90)
                # SFX: Level_Complete
            else:
                self.score -= 50
                self._set_message("OUT OF ACTIVATIONS", 90)
                # SFX: Failure
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        self.maze_width = 31
        self.maze_height = 15
        self.cell_size = self.GAME_AREA_HEIGHT // self.maze_height
        
        self.portal_activations_left = 3 + (self.level -1) // 5
        self.portal_activations_left = min(self.portal_activations_left, 10)

        # Generate maze and place start/exit
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.start_pos = (1, 1)
        self.exit_pos = (self.maze_width - 2, self.maze_height - 2)
        self.maze[self.exit_pos[1], self.exit_pos[0]] = 0 # Ensure exit is open
        self.maze[self.start_pos[1], self.start_pos[0]] = 0 # Ensure start is open

        self.player_pos = np.array(self.start_pos, dtype=float)
        self.player_visual_pos = self.player_pos.copy()

        # Select word pairs and associate them with morphable walls
        num_pairs = self.portal_activations_left
        available_pairs = random.sample(self.WORD_PAIRS_MASTER, num_pairs)
        self.words = []
        self.word_map = {}
        
        dead_ends = self._find_dead_ends()
        random.shuffle(dead_ends)

        for i, pair in enumerate(available_pairs):
            self.words.extend(pair)
            if i < len(dead_ends):
                wall_to_morph = dead_ends[i]
                self.maze[wall_to_morph[1], wall_to_morph[0]] = 2 # Mark as morphable
                self.word_map[tuple(sorted(pair))] = wall_to_morph
        
        random.shuffle(self.words)
        self.word_cursor_idx = 0
        self.selected_indices = []

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=int)
        start_x, start_y = (1, 1)
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < width -1 and 0 < ny < height -1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[ny, nx] = 0
                maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_dead_ends(self):
        dead_ends = []
        for y in range(1, self.maze_height - 1):
            for x in range(1, self.maze_width - 1):
                if self.maze[y, x] == 1: # Is a wall
                    # Check if it connects two paths
                    paths = 0
                    if self.maze[y-1, x] == 0: paths += 1
                    if self.maze[y+1, x] == 0: paths += 1
                    if self.maze[y, x-1] == 0: paths += 1
                    if self.maze[y, x+1] == 0: paths += 1
                    if paths >= 2:
                        dead_ends.append((x, y))
        return dead_ends

    def _move_player(self, movement):
        target_pos = self.player_pos.copy()
        if movement == 1: target_pos[1] -= 1  # Up
        elif movement == 2: target_pos[1] += 1  # Down
        elif movement == 3: target_pos[0] -= 1  # Left
        elif movement == 4: target_pos[0] += 1  # Right
        
        tx, ty = int(target_pos[0]), int(target_pos[1])
        if 0 <= tx < self.maze_width and 0 <= ty < self.maze_height and self.maze[ty, tx] != 1:
            self.player_pos = target_pos
            # SFX: Player_Move_Step

    def _attempt_word_match(self):
        word1 = self.words[self.selected_indices[0]]
        word2 = self.words[self.selected_indices[1]]
        
        pair_key = tuple(sorted((word1, word2)))
        
        if pair_key in self.word_map and self.portal_activations_left > 0:
            wall_pos = self.word_map[pair_key]
            self.maze[wall_pos[1], wall_pos[0]] = 0 # Morph the wall
            self.portal_activations_left -= 1
            del self.word_map[pair_key]
            self._set_message("PATH OPENED", 60)
            self._create_ripple_effect(wall_pos)
            # SFX: Match_Success
            reward = 10
        else:
            self._set_message("MISMATCH", 60)
            # SFX: Match_Fail
            reward = -1
            
        self.selected_indices = []
        return reward

    def _calculate_distance_to_exit(self):
        return abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])

    def _check_termination(self):
        at_exit = self.player_pos[0] == self.exit_pos[0] and self.player_pos[1] == self.exit_pos[1]
        no_more_moves = self.portal_activations_left <= 0 and not any(self.maze.flatten() == 2)
        
        # Check if exit is reachable with current maze state
        if no_more_moves and not at_exit:
            if not self._is_exit_reachable():
                return True

        return at_exit or self.steps >= self.MAX_STEPS

    def _is_exit_reachable(self):
        q = [tuple(self.player_pos.astype(int))]
        visited = {tuple(self.player_pos.astype(int))}
        while q:
            x, y = q.pop(0)
            if (x, y) == self.exit_pos:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.maze_width and 0 <= ny < self.maze_height and self.maze[ny, nx] != 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _get_observation(self):
        # --- Update visual state ---
        lerp_factor = 0.5
        self.player_visual_pos = self.player_visual_pos * (1 - lerp_factor) + self.player_pos * lerp_factor

        # --- Render all elements ---
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_maze()
        self._render_exit()
        self._render_player()
        self._render_effects()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), but gym expects (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "total_score": self.total_score,
            "steps": self.steps,
            "level": self.level,
            "activations_left": self.portal_activations_left,
        }

    def _create_gradient_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_START[0] * (1 - ratio) + self.COLOR_BG_END[0] * ratio),
                int(self.COLOR_BG_START[1] * (1 - ratio) + self.COLOR_BG_END[1] * ratio),
                int(self.COLOR_BG_START[2] * (1 - ratio) + self.COLOR_BG_END[2] * ratio),
            )
            pygame.draw.line(bg, color, (0, y), (self.SCREEN_WIDTH, y))
        return bg

    def _render_maze(self):
        offset_x = (self.SCREEN_WIDTH - self.maze_width * self.cell_size) // 2
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                rect = pygame.Rect(offset_x + x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif self.maze[y, x] == 2:
                    pygame.draw.rect(self.screen, self.COLOR_WALL_MORPHABLE, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GROUND, rect)

    def _render_player(self):
        offset_x = (self.SCREEN_WIDTH - self.maze_width * self.cell_size) // 2
        px = int(offset_x + self.player_visual_pos[0] * self.cell_size + self.cell_size / 2)
        py = int(self.player_visual_pos[1] * self.cell_size + self.cell_size / 2)
        radius = int(self.cell_size * 0.35)
        
        # Glow effect
        for i in range(radius, 0, -2):
            alpha = int(self.COLOR_PLAYER_GLOW[3] * (i / radius)**2)
            color = self.COLOR_PLAYER_GLOW[:3] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, px, py, i + radius // 2, color)
        
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)

    def _render_exit(self):
        offset_x = (self.SCREEN_WIDTH - self.maze_width * self.cell_size) // 2
        ex = int(offset_x + self.exit_pos[0] * self.cell_size + self.cell_size / 2)
        ey = int(self.exit_pos[1] * self.cell_size + self.cell_size / 2)
        radius = int(self.cell_size * 0.4)

        # Glow
        for i in range(radius, 0, -2):
            alpha = int(self.COLOR_EXIT_GLOW[3] * (i / radius)**1.5)
            color = self.COLOR_EXIT_GLOW[:3] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, i + radius // 2, color)

        pygame.gfxdraw.aacircle(self.screen, ex, ey, radius, self.COLOR_EXIT)
        pygame.gfxdraw.filled_circle(self.screen, ex, ey, radius, self.COLOR_EXIT)

    def _render_ui(self):
        # --- Top UI Bar ---
        level_text = self.font_ui.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        activations_text = self.font_ui.render(f"Activations: {self.portal_activations_left}", True, self.COLOR_TEXT)
        
        self.screen.blit(level_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))
        self.screen.blit(activations_text, (self.SCREEN_WIDTH - activations_text.get_width() - 10, 10))

        # --- Bottom Word Bar ---
        bar_y = self.GAME_AREA_HEIGHT + 10
        total_word_width = sum(self.font_words.render(w, True, self.COLOR_TEXT).get_width() for w in self.words) + (len(self.words) - 1) * 20
        start_x = (self.SCREEN_WIDTH - total_word_width) // 2
        current_x = start_x

        for i, word in enumerate(self.words):
            color = self.COLOR_TEXT
            if i in self.selected_indices:
                color = self.COLOR_TEXT_SELECTED
            elif i == self.word_cursor_idx:
                color = self.COLOR_TEXT_HIGHLIGHT
            
            word_surf = self.font_words.render(word, True, color)
            self.screen.blit(word_surf, (current_x, bar_y + (30 - word_surf.get_height())//2))

            if i == self.word_cursor_idx:
                underline_y = bar_y + 35
                pygame.draw.line(self.screen, self.COLOR_TEXT_HIGHLIGHT, (current_x, underline_y), (current_x + word_surf.get_width(), underline_y), 2)

            current_x += word_surf.get_width() + 20

        # --- Message Display ---
        if self.message_timer > 0:
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT_HIGHLIGHT)
            pos = (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_surf.get_height() // 2)
            self.screen.blit(msg_surf, pos)

    def _set_message(self, text, duration):
        self.message = text
        self.message_timer = duration

    def _create_ripple_effect(self, grid_pos):
        offset_x = (self.SCREEN_WIDTH - self.maze_width * self.cell_size) // 2
        px = int(offset_x + grid_pos[0] * self.cell_size + self.cell_size / 2)
        py = int(grid_pos[1] * self.cell_size + self.cell_size / 2)
        self.ripple_effects.append({'pos': (px, py), 'radius': 0, 'max_radius': 80, 'life': 60})

    def _update_effects(self):
        if self.message_timer > 0:
            self.message_timer -= 1
        
        for ripple in self.ripple_effects[:]:
            ripple['life'] -= 1
            if ripple['life'] <= 0:
                self.ripple_effects.remove(ripple)
            else:
                ripple['radius'] += ripple['max_radius'] / 60

    def _render_effects(self):
        for ripple in self.ripple_effects:
            progress = ripple['radius'] / ripple['max_radius']
            alpha = int(150 * (1 - progress)**2)
            if alpha > 0:
                color = self.COLOR_EXIT + (alpha,)
                pygame.gfxdraw.aacircle(self.screen, ripple['pos'][0], ripple['pos'][1], int(ripple['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, ripple['pos'][0], ripple['pos'][1], int(ripple['radius']-2), color)

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dream Maze")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset level")
    print("Q: Quit")
    print("----------------\n")

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset(options={'level': env.level})
            
        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Screen ---
        # The observation is already the rendered screen, just need to transpose it back
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        game_window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final Score: {info['score']}. Total Score: {info['total_score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            if env.player_pos[0] == env.exit_pos[0] and env.player_pos[1] == env.exit_pos[1]:
                 obs, info = env.reset(options={'level': env.level}) # Progress to next level
            else:
                 obs, info = env.reset(options={'level': 1}) # Reset from level 1 on loss
                 env.total_score = 0


        clock.tick(30) # Run at 30 FPS

    env.close()