
# Generated: 2025-08-28T01:58:16.373845
# Source Brief: brief_04290.md
# Brief Index: 4290

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a number and place it in the equation. Shift to clear the equation."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Crack the code! Move numbers from the grid to solve the target equation before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 4
        self.CELL_SIZE = 70
        self.GRID_MARGIN_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_MARGIN_Y = 20
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (44, 62, 80) # Dark blue-grey
        self.COLOR_GRID = (52, 73, 94)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_CURSOR = (241, 196, 15) # Yellow
        self.COLOR_CURSOR_GLOW = (241, 196, 15, 50)
        self.COLOR_EMPTY_SLOT = (127, 140, 141)
        self.COLOR_WIN = (46, 204, 113)
        self.COLOR_LOSE = (231, 76, 60)
        
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
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 30)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.equation_slots = None
        self.solution = None
        self.moves_left = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.flash_effect = None

        # Initialize state
        self.reset()

        # Run validation
        self.validate_implementation()
    
    def _generate_puzzle(self):
        """Generates a new solvable puzzle."""
        while True:
            # Generate a grid of unique numbers
            all_numbers = list(range(1, 26))
            self.np_random.shuffle(all_numbers)
            grid_numbers = all_numbers[:self.GRID_SIZE**2]
            
            self.grid = np.array(grid_numbers).reshape((self.GRID_SIZE, self.GRID_SIZE))

            # Find a valid equation (a + b = c) within the grid
            possible_solutions = []
            for i in range(len(grid_numbers)):
                for j in range(i + 1, len(grid_numbers)):
                    a, b = grid_numbers[i], grid_numbers[j]
                    if a + b in grid_numbers:
                        possible_solutions.append(sorted([a, b]) + [a + b])
            
            if possible_solutions:
                self.solution = self.np_random.choice(possible_solutions)
                return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_puzzle()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.equation_slots = [None, None, None] # Stores (number, (orig_r, orig_c))
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.flash_effect = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action[0], action[1], action[2]
        space_pressed = space_val == 1 and not self.last_space_held
        shift_pressed = shift_val == 1 and not self.last_shift_held
        self.last_space_held = (space_val == 1)
        self.last_shift_held = (shift_val == 1)

        reward = 0
        action_taken = False

        # 1. Handle Movement
        if movement > 0:
            r, c = self.cursor_pos
            if movement == 1 and r > 0: self.cursor_pos[0] -= 1
            elif movement == 2 and r < self.GRID_SIZE - 1: self.cursor_pos[0] += 1
            elif movement == 3 and c > 0: self.cursor_pos[1] -= 1
            elif movement == 4 and c < self.GRID_SIZE - 1: self.cursor_pos[1] += 1
            
            if self.cursor_pos != [r, c]:
                action_taken = True
        
        # 2. Handle Shift (Clear Equation)
        if shift_pressed and any(self.equation_slots):
            self._clear_equation()
            action_taken = True
            # sfx: UI_deselect

        # 3. Handle Space (Place Number)
        if space_pressed:
            r, c = self.cursor_pos
            num_to_place = self.grid[r, c]
            
            try:
                next_slot_idx = [s is None for s in self.equation_slots].index(True)
                if num_to_place is not None:
                    self.equation_slots[next_slot_idx] = (num_to_place, (r, c))
                    self.grid[r, c] = 0 # Use 0 as a sentinel for 'empty'
                    action_taken = True
                    self._create_particles(c, r)
                    # sfx: place_number
            except ValueError:
                # Equation is full, do nothing
                pass

        # 4. Update state if an action was taken
        if action_taken:
            self.moves_left -= 1
            reward = -0.1

        # 5. Check for equation completion and correctness
        if None not in self.equation_slots:
            nums = [s[0] for s in self.equation_slots]
            # Check a+b=c or b+a=c
            is_correct = (nums[0] + nums[1] == nums[2]) or (nums[1] + nums[0] == nums[2])
            
            if is_correct:
                self.win_state = True
                self.game_over = True
                reward += 100
                self.score += 100
                self.flash_effect = (self.COLOR_WIN, 30) # color, duration
                # sfx: win_sound
            else:
                reward += -10
                self.score -= 10
                self._clear_equation()
                self.flash_effect = (self.COLOR_LOSE, 30)
                # sfx: wrong_guess

        # 6. Check for termination conditions
        self.steps += 1
        terminated = self.moves_left <= 0 or self.win_state or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if not self.win_state:
                reward += -20
                self.score -= 20
                self.flash_effect = (self.COLOR_LOSE, 30)
                # sfx: lose_sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _clear_equation(self):
        """Puts numbers from equation slots back onto the grid."""
        for slot in self.equation_slots:
            if slot is not None:
                num, (r, c) = slot
                self.grid[r, c] = num
        self.equation_slots = [None, None, None]

    def _create_particles(self, grid_c, grid_r):
        """Create a burst of particles."""
        cell_center_x = self.GRID_MARGIN_X + (grid_c + 0.5) * self.CELL_SIZE
        cell_center_y = self.GRID_MARGIN_Y + (grid_r + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append([cell_center_x, cell_center_y, dx, dy, life, self.COLOR_CURSOR])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[1] # x += dx
            p[1] += p[2] # y += dy
            p[4] -= 1 # life -= 1
            if p[4] > 0:
                alpha = max(0, min(255, int(255 * (p[4] / 40))))
                size = max(1, int(p[4] / 8))
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, p[5] + (alpha,))
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid and numbers
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.CELL_SIZE,
                    self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2)
                
                num = self.grid[r, c]
                if num != 0:
                    text_surf = self.font_large.render(str(num), True, self.COLOR_TEXT)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
                else: # Empty spot
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 10, self.COLOR_GRID)

        # Draw cursor
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + c * self.CELL_SIZE,
            self.GRID_MARGIN_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Glow effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
        for i in range(4):
            glow_alpha = int(80 * (1 - i/4) * pulse)
            glow_color = self.COLOR_CURSOR + (glow_alpha,)
            pygame.draw.rect(self.screen, glow_color, cursor_rect.inflate(i*4, i*4), 2, border_radius=5)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=5)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Draw equation
        eq_y = self.HEIGHT - 60
        
        num1 = str(self.equation_slots[0][0]) if self.equation_slots[0] else "_"
        num2 = str(self.equation_slots[1][0]) if self.equation_slots[1] else "_"
        num3 = str(self.equation_slots[2][0]) if self.equation_slots[2] else "_"
        
        eq_str = f"{num1}  +  {num2}  =  {num3}"
        eq_surf = self.font_large.render(eq_str, True, self.COLOR_TEXT)
        eq_rect = eq_surf.get_rect(center=(self.WIDTH // 2, eq_y))
        self.screen.blit(eq_surf, eq_rect)

        # Draw info text (moves and score)
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (20, self.HEIGHT - 40))

        score_text = f"Score: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(right=self.WIDTH - 20, top=self.HEIGHT - 40)
        self.screen.blit(score_surf, score_rect)

        # Handle flash effect
        if self.flash_effect:
            color, duration = self.flash_effect
            if duration > 0:
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                alpha = int(200 * (duration / 30))
                s.fill(color + (alpha,))
                self.screen.blit(s, (0, 0))
                self.flash_effect = (color, duration - 1)
            else:
                self.flash_effect = None

        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                msg = "SOLVED!"
                color = self.COLOR_WIN
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "solution": self.solution.tolist() if self.solution is not None else [],
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Code Cracker")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            action = [movement, space_held, shift_held]
            
            # This is a turn-based game, so we only step on an input
            if movement != 0 or space_held != env.last_space_held or shift_held != env.last_shift_held:
                 obs, reward, terminated, truncated, info = env.step(action)
                 print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

            if terminated:
                print("Game Over!")
                print(f"Final Score: {info['score']}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()