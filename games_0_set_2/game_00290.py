
# Generated: 2025-08-27T13:12:43.007823
# Source Brief: brief_00290.md
# Brief Index: 290

        
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


# A small, embedded word list for the game.
# This avoids external file dependencies.
WORD_LIST_STR = """
able acid aged also area army away baby back ball band bank base bath bear beat
bell belt best bill bird blow blue boat body bomb bond bone book boom born boss
both bowl bridge brief bring broad broke brown build bulk burn bush busy call
calm came camp card care case cast cell cent city club coal coat code cold come
cook cool cope core cost crew crop cross dark data date dead deal dear debt deep
deny desk dial die a diet dirt dish disk does done door dose down draw drew drop
drug drum dual duck due dull dust duty each earn ease east easy edge else even
ever evil exit face fact fail fair fall farm fast fate fear feed feel fell felt
fill film find fine fire firm fish five flat flow food foot form fort four free
from front fuel full fund fuse gain game gate gave gear gene get girl give glad
goes gold good got gray grew grey grip grow gulf half hall hand hang hard harm
have head hear heat heavy help high hill hire hold hole home hope horn host hour
how huge hunt hurt idea idle iron item join joke judge jury just keep keen kept
key kick kill kind king knee knew know lack lady laid lake land lane large last
late lead left less life lift like line link list live load loan lock long look
lose loss lost loud love low luck lunch made mail main make male many mark mass
mast mate meal mean meat meet melt mere mild mile milk mill mind mine miss mode
mood moon more most move much must name navy near neck need new next nice nine
none noon norm nose note nuke null numb obey object odd off oil old only onto
open oral order other our out over own pace pack page paid pain pair palm panel
park part pass past path peak pick pile pink pipe plan play plot plug plus poll
pool poor port post pull pump push put race rail rain rank rare rate raw reach
read real rear rely rent rest rich ride ring rise risk road rock role roll room
root rope rose rough round row ruby rude rule rush safe said sail sale salt same
sand save say scale scope score seal seat see seek seem seen self sell send sense
sent serve seven shed ship shoe shop shot show sick side sign silk since sing
size skin slab slow small smart smell smile smoke snow soft solid some song soon
sore sort soul sound south space spare speak speed spend spent spin spot spread
spring squad square stack staff stage stand star start state stay steel stem
step stick still stock stone stood stop store storm story strip stuck study stuff
such suit sun sure take tale talk tall tank tape task team tear term test than
that them then they thin this thou three throw thus tide tidy tie time tiny told
tone tool top total touch tough tour town track trade trail train trap tree trip
true trust try tube tune turn twin type unit upon used user usual value vast
very vice view visit vital voice vote wage wait wake walk wall want ward warm
warn wash waste watch water wave way weak wear week well went were west what
when where which while white who whole why wide wife will wind wine wing wipe
wire wise wish with within wood word work world worm worn worry worth would
wound wrap write yard year yell yet you your zero zone
"""
WORD_SET = set(WORD_LIST_STR.strip().split())

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to select adjacent letters. Space to submit a word. Shift to clear your selection."
    )

    game_description = (
        "Form words from adjacent letter blocks to break them and clear the grid in this fast-paced arcade puzzle."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_block = pygame.font.Font(None, 48)
        self.font_ui = pygame.font.Font(None, 28)

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_SIZE = 40
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_Y = 60

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_SELECT = (255, 80, 80, 150)
        self.COLOR_INVALID = (255, 0, 0)
        self.LETTER_COLORS = {
            1: (180, 220, 255), 2: (160, 255, 160), 3: (255, 240, 160),
            4: (255, 180, 160), 5: (220, 160, 255), 8: (255, 160, 220), 10: (255, 120, 120)
        }
        self.LETTER_VALUES = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
            'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
            'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }
        self.VOWELS = "AEIOU"
        self.CONSONANTS = "BCDFGHJKLMNPQRSTVWXYZ"

        # --- State Variables (initialized in reset) ---
        self.grid = []
        self.cursor_pos = (0, 0)
        self.selection_path = []
        self.current_word = ""
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.word_feedback_timer = 0
        self.word_feedback_text = ""
        self.word_feedback_color = self.COLOR_TEXT

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 15
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.selection_path = []
        self.current_word = ""
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.word_feedback_timer = 0

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Word Feedback Animation ---
        if self.word_feedback_timer > 0:
            self.word_feedback_timer -= 1

        # --- Handle Actions ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if shift_pressed:
            # sfx: clear_selection_sound
            self.selection_path = []
            self.current_word = ""

        elif space_pressed and self.selection_path:
            self.moves_left -= 1
            word_is_valid = self.current_word.lower() in WORD_SET and len(self.current_word) > 1

            if word_is_valid:
                # sfx: valid_word_sound
                num_cleared = len(self.selection_path)
                reward += num_cleared  # +1 per block
                word_score = sum(self.grid[y][x]['value'] for x, y in self.selection_path)
                self.score += word_score

                if num_cleared > 3:
                    reward += 5  # Bonus for long words

                for x, y in self.selection_path:
                    self._create_particles(x, y)
                    self.grid[y][x] = None
                
                self._apply_gravity_and_refill()
                self.word_feedback_text = f"'{self.current_word}' +{word_score}"
                self.word_feedback_color = (150, 255, 150)
                self.word_feedback_timer = 30 # frames

            else:
                # sfx: invalid_word_sound
                self.word_feedback_text = f"'{self.current_word}' - Not a word!"
                self.word_feedback_color = self.COLOR_INVALID
                self.word_feedback_timer = 30 # frames

            self.selection_path = []
            self.current_word = ""

        elif movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = (self.cursor_pos[0] + dx)
            new_y = (self.cursor_pos[1] + dy)

            # Clamp to grid boundaries
            new_x = max(0, min(self.GRID_COLS - 1, new_x))
            new_y = max(0, min(self.GRID_ROWS - 1, new_y))
            
            # Start a new selection if path is empty
            if not self.selection_path:
                self.selection_path.append((self.cursor_pos[0], self.cursor_pos[1]))
            
            self.cursor_pos = (new_x, new_y)
            
            # Add to path if not already present
            if (new_x, new_y) not in self.selection_path:
                # sfx: select_letter_sound
                self.selection_path.append((new_x, new_y))
            # Or, if it's the second to last, unselect the last one (backtracking)
            elif len(self.selection_path) > 1 and (new_x, new_y) == self.selection_path[-2]:
                # sfx: unselect_letter_sound
                self.selection_path.pop()

        # Update current word string from path
        self.current_word = "".join(self.grid[y][x]['letter'] for x, y in self.selection_path if self.grid[y][x])
        
        self.steps += 1
        terminated = self._check_termination()
        
        # --- Terminal Rewards ---
        if terminated:
            if all(cell is None for row in self.grid for cell in row): # Win
                reward += 50
            elif self.moves_left <= 0: # Lose
                reward -= 50
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Win condition: all blocks cleared
        if all(cell is None for row in self.grid for cell in row):
            self.game_over = True
            return True

        # Loss condition: no moves left
        if self.moves_left <= 0:
            self.game_over = True
            return True
            
        # Step limit
        if self.steps >= 1000:
            self.game_over = True
            return True

        return False

    def _generate_grid(self):
        self.grid = [[self._random_letter_block() for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        # Simple guarantee of playability: ensure at least some vowels exist
        for _ in range(self.GRID_COLS):
            rx, ry = random.randint(0, self.GRID_COLS - 1), random.randint(0, self.GRID_ROWS - 1)
            letter = random.choice(self.VOWELS)
            self.grid[ry][rx] = {'letter': letter, 'value': self.LETTER_VALUES[letter]}


    def _random_letter_block(self):
        # Weighted choice for more common letters
        if random.random() < 0.6: # 60% chance of being a common letter
             letter = random.choice("EARIOTNSLC")
        else:
             letter = random.choice(self.VOWELS + self.CONSONANTS)
        return {'letter': letter, 'value': self.LETTER_VALUES[letter]}

    def _apply_gravity_and_refill(self):
        for col in range(self.GRID_COLS):
            empty_slots = 0
            for row in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[row][col] is None:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[row + empty_slots][col] = self.grid[row][col]
                    self.grid[row][col] = None
            
            # Refill top rows
            for i in range(empty_slots):
                self.grid[i][col] = self._random_letter_block()

    def _create_particles(self, grid_x, grid_y):
        px = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        
        block = self.grid[grid_y][grid_x]
        if not block: return

        color_key = min(self.LETTER_COLORS.keys(), key=lambda k: abs(k-block['value']))
        color = self.LETTER_COLORS[color_key]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.uniform(3, 7)
            life = random.randint(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'size': size, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['size'] -= 0.1
            p['life'] -= 1
            if p['size'] <= 0 or p['life'] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], int(p['size']))

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw blocks and letters
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                block = self.grid[y][x]
                if block:
                    rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    
                    # Draw block background color based on value
                    color_key = min(self.LETTER_COLORS.keys(), key=lambda k: abs(k-block['value']))
                    block_color = self.LETTER_COLORS[color_key]
                    pygame.draw.rect(self.screen, block_color, rect.inflate(-4, -4), border_radius=4)
                    
                    # Draw letter
                    letter_surf = self.font_block.render(block['letter'], True, self.COLOR_BG)
                    letter_rect = letter_surf.get_rect(center=rect.center)
                    self.screen.blit(letter_surf, letter_rect)

        # Draw selection overlay
        for i, (x, y) in enumerate(self.selection_path):
            rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            
            # Draw line to previous block
            if i > 0:
                prev_x, prev_y = self.selection_path[i-1]
                start_pos = (self.GRID_X + prev_x * self.CELL_SIZE + self.CELL_SIZE//2, self.GRID_Y + prev_y * self.CELL_SIZE + self.CELL_SIZE//2)
                end_pos = (rect.centerx, rect.centery)
                pygame.draw.line(self.screen, self.COLOR_SELECT, start_pos, end_pos, 5)

            # Draw selection highlight
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECT)
            self.screen.blit(s, rect.topleft)

        # Draw cursor
        cursor_rect = pygame.Rect(self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=4)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Current word display
        word_surf = self.font_main.render(f"Word: {self.current_word}", True, self.COLOR_TEXT)
        self.screen.blit(word_surf, (self.GRID_X, 20))

        # Score display
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_surf, score_rect)
        
        # Moves display
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 35))
        self.screen.blit(moves_surf, moves_rect)

        # Word submission feedback
        if self.word_feedback_timer > 0:
            alpha = int(255 * (self.word_feedback_timer / 30))
            feedback_surf = self.font_main.render(self.word_feedback_text, True, self.word_feedback_color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.GRID_Y + self.GRID_ROWS * self.CELL_SIZE + 25))
            self.screen.blit(feedback_surf, feedback_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            win = all(cell is None for row in self.grid for cell in row)
            msg = "GRID CLEARED!" if win else "GAME OVER"
            color = (150, 255, 150) if win else (255, 100, 100)
            
            end_surf = self.font_main.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_surf, end_rect)

            score_surf = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(score_surf, score_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by an RL agent
    
    # Set this to True to run an automated test, False to play manually
    automated_test = False

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    if not automated_test:
        pygame.display.set_caption("Word Grid")
        screen = pygame.display.set_mode((640, 400))
        
        while not done:
            movement = 0 # no-op
            space = 0
            shift = 0
            
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    action_taken = True
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                    elif event.key == pygame.K_SPACE:
                        space = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift = 1
                    elif event.key == pygame.K_r: # Reset button
                        obs, info = env.reset()
                    else:
                        action_taken = False

            action = [movement, space, shift]
            
            # Since auto_advance is False, we only step on an action
            if action_taken:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated
                print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_left']}, Done: {done}")
            else: # Still need to render if no action
                obs = env._get_observation()

            # Draw the observation to the display screen
            draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(draw_surface, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit frame rate
            
        env.close()

    # --- Automated Test ---
    else:
        print("Running automated test...")
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if (i % 100 == 0):
                print(f"Step {i}: Reward={reward}, Info={info}")
            if terminated:
                print(f"Episode finished after {i+1} steps.")
                obs, info = env.reset()
        env.close()
        print("Automated test complete.")