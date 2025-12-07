
# Generated: 2025-08-27T13:17:04.445165
# Source Brief: brief_00312.md
# Brief Index: 312

        
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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move cursor. Hold Space and drag to select letters, release to submit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find and form words on a grid by connecting adjacent letters. "
        "Clear 50% of the board in 10 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 5
    CELL_SIZE = 70
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2 + 20

    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINE = (50, 60, 70)
    COLOR_EMPTY_CELL = (30, 40, 50)
    COLOR_LETTER = (230, 240, 255)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_PATH = (0, 150, 255, 180)  # RGBA for transparency
    COLOR_VALID = (0, 255, 100)
    COLOR_INVALID = (255, 80, 80)
    COLOR_UI_TEXT = (200, 200, 220)

    VOWELS = "AEIOU"
    CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"
    WORD_LIST = {
        "cat", "dog", "sun", "run", "big", "red", "bed", "egg", "fun", "get", "hat", "jet", "let", "man", "net", "pen", "rat", "sit", "ten", "vet", "wet", "yet", "zen",
        "acid", "bake", "call", "dark", "each", "face", "game", "hail", "icon", "jade", "keen", "lake", "make", "name", "oath", "pace", "quad", "race", "safe", "take", "unit", "vale", "walk", "zone",
        "above", "basic", "cable", "daily", "early", "faith", "giant", "habit", "ideal", "joint", "karma", "label", "magic", "naval", "ocean", "paint", "quake", "raise", "saint", "table", "ultra", "value", "waste", "yield",
        "abroad", "beacon", "calmly", "damage", "eagerly", "fabric", "galaxy", "hacker", "impact", "jacket", "keeper", "ladder", "magnet", "narrow", "object", "palace", "quarry", "radial", "safety", "tackle", "umpire", "vacant", "wander",
        "ability", "backing", "cabinet", "dancing", "eastern", "factory", "gallery", "habitat", "iceberg", "journey", "kitchen", "landing", "machine", "natural", "oatmeal", "package", "quality", "radical", "sailing", "tactful", "uncover", "vaccine", "walking",
        "absolute", "bachelor", "capacity", "database", "economic", "facility", "generate", "handling", "identify", "judicial", "keyboard", "language", "maintain", "national", "obedient", "painting", "quantity", "railroad", "sampling", "tactical", "ultimate", "validate"
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_letter = pygame.font.Font(None, 50)
        self.font_ui = pygame.font.Font(None, 32)
        self.font_feedback = pygame.font.Font(None, 48)
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.is_selecting = False
        self.current_path = []
        self.last_space_held = False
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.initial_tile_count = 0
        self.feedback_message = ""
        self.feedback_color = (0,0,0)
        self.feedback_timer = 0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 10
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.is_selecting = False
        self.current_path = []
        self.last_space_held = False
        self.feedback_message = ""
        self.feedback_timer = 0

        self._generate_grid()
        self.initial_tile_count = self.GRID_SIZE * self.GRID_SIZE

        return self._get_observation(), self._get_info()
    
    def _generate_grid(self):
        self.grid = []
        for _ in range(self.GRID_SIZE):
            row = []
            for _ in range(self.GRID_SIZE):
                if self.np_random.random() < 0.4: # 40% chance of a vowel
                    row.append(self.np_random.choice(list(self.VOWELS)))
                else:
                    row.append(self.np_random.choice(list(self.CONSONANTS)))
            self.grid.append(row)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_message = ""

        self._move_cursor(movement)

        if space_held and not self.last_space_held:
            if not self.is_selecting:
                cy, cx = self.cursor_pos
                if self.grid[cy][cx] != ' ':
                    # sfx: start_selection_sound
                    self.is_selecting = True
                    self.current_path = [(cy, cx)]
        
        if not space_held and self.last_space_held:
            if self.is_selecting:
                reward = self._submit_word()

        self.last_space_held = space_held
        
        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        cy, cx = self.cursor_pos
        prev_pos = tuple(self.cursor_pos)

        if movement == 1: cy -= 1
        elif movement == 2: cy += 1
        elif movement == 3: cx -= 1
        elif movement == 4: cx += 1
        
        self.cursor_pos = [np.clip(cy, 0, self.GRID_SIZE - 1), np.clip(cx, 0, self.GRID_SIZE - 1)]
        new_pos = tuple(self.cursor_pos)

        if self.is_selecting and new_pos != prev_pos:
            last_pos = self.current_path[-1]
            is_adjacent = abs(new_pos[0] - last_pos[0]) <= 1 and abs(new_pos[1] - last_pos[1]) <= 1
            is_new = new_pos not in self.current_path
            
            if is_adjacent and is_new and self.grid[new_pos[0]][new_pos[1]] != ' ':
                # sfx: add_letter_sound
                self.current_path.append(new_pos)

    def _submit_word(self):
        reward = 0
        self.moves_left -= 1
        
        word = "".join([self.grid[r][c] for r, c in self.current_path]).lower()
        
        if len(word) >= 3 and word in self.WORD_LIST:
            # sfx: valid_word_sound
            reward += 0.1 + len(word)
            self.score += len(word) * 10
            
            for r, c in self.current_path:
                self.grid[r][c] = ' '
            
            self.feedback_message = f"'{word.upper()}' +{len(word) * 10}"
            self.feedback_color = self.COLOR_VALID
            self.feedback_timer = 60
        else:
            # sfx: invalid_word_sound
            reward -= 0.1
            self.feedback_message = "INVALID"
            self.feedback_color = self.COLOR_INVALID
            self.feedback_timer = 45
            
        self.is_selecting = False
        self.current_path = []

        if self._check_win_condition():
            reward += 50
            self.score += 1000
            self.feedback_message = "GRID CLEARED!"
            self.feedback_color = self.COLOR_VALID
            self.feedback_timer = 90
        
        return reward

    def _check_win_condition(self):
        cleared_count = sum(row.count(' ') for row in self.grid)
        return cleared_count >= self.initial_tile_count * 0.5

    def _check_termination(self):
        if self.game_over:
            return True
        
        win = self._check_win_condition()
        lose = self.moves_left <= 0
        timeout = self.steps >= 1000

        if win or lose or timeout:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _render_game(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.CELL_SIZE,
                    self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_EMPTY_CELL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

                letter = self.grid[r][c]
                if letter != ' ':
                    letter_surf = self.font_letter.render(letter, True, self.COLOR_LETTER)
                    letter_rect = letter_surf.get_rect(center=rect.center)
                    self.screen.blit(letter_surf, letter_rect)

        if self.is_selecting and self.current_path:
            path_points = [
                (self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                 self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2)
                for r, c in self.current_path
            ]
            if len(path_points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_PATH[:3], False, path_points, 8)
            for r, c in self.current_path:
                highlight_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                highlight_surf.fill(self.COLOR_PATH)
                self.screen.blit(highlight_surf, (self.GRID_MARGIN_X + c * self.CELL_SIZE, self.GRID_MARGIN_Y + r * self.CELL_SIZE))

        cy, cx = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + cx * self.CELL_SIZE, self.GRID_MARGIN_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=4)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 15))

        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(moves_surf, moves_rect)

        if self.feedback_timer > 0:
            feedback_surf = self.font_feedback.render(self.feedback_message, True, self.feedback_color)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.GRID_MARGIN_Y // 2))
            self.screen.blit(feedback_surf, feedback_rect)
            
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self._check_win_condition() else "GAME OVER"
            end_color = self.COLOR_VALID if self._check_win_condition() else self.COLOR_INVALID
            
            end_surf = self.font_feedback.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(end_surf, end_rect)

            final_score_text = f"Final Score: {self.score}"
            final_score_surf = self.font_ui.render(final_score_text, True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 30))
            self.screen.blit(final_score_surf, final_score_rect)


    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will open a window and accept keyboard inputs
    
    # Re-initialize Pygame for a visible display
    pygame.quit()
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    # Create the environment
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      Word Grid - Human Player")
    print("="*30)
    print(env.user_guide)
    print("Press 'R' to reset the game.\n")
    
    # Game loop
    running = True
    while running:
        movement = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
        
        # Get keyboard state for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment if the game is not over
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the current observation to the screen
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # Maintain a 30 FPS frame rate
        clock.tick(30)

    env.close()
    pygame.quit()