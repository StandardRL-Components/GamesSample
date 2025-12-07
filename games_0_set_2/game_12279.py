import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:38:10.570752
# Source Brief: brief_02279.md
# Brief Index: 2279
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a roguelike puzzle game where the player
    builds programs from letters to terraform a digital landscape.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A roguelike puzzle game. Assemble programs from a pool of letters to terraform a digital landscape and solve challenges."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to pick up or drop a letter. Press shift to place a letter into the program."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (40, 45, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_GLOW = (255, 255, 0, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_STABLE = (20, 180, 120)
    COLOR_STABLE_EDGE = (10, 120, 80)
    COLOR_UNSTABLE = (220, 50, 50)
    COLOR_UNSTABLE_EDGE = (150, 30, 30)
    COLOR_EMPTY = (25, 30, 50)
    COLOR_MAGNETIZED = (80, 150, 255)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 20, 8
    CELL_SIZE = 32
    GRID_OFFSET_X = 0
    GRID_OFFSET_Y = SCREEN_HEIGHT - (GRID_ROWS * CELL_SIZE)

    # UI Layout
    UI_TOP_BAR_H = 40
    UI_LETTER_POOL_H = 60
    UI_PROGRAM_AREA_Y = UI_TOP_BAR_H
    UI_LETTER_POOL_Y = UI_TOP_BAR_H + UI_PROGRAM_AREA_Y

    # Game Mechanics
    MAX_STEPS = 2000
    CURSOR_SPEED = 1 # cells per step
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_code = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_small = pygame.font.Font(None, 20)

        # --- Game State Initialization ---
        self.level = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.landscape_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        
        # Cursor is on a logical grid that spans all interactive areas
        self.cursor_grid_width = self.GRID_COLS
        self.cursor_grid_height = 2 + self.GRID_ROWS # 1 for program, 1 for pool
        self.cursor_pos = [0, 0] # [col, row]
        
        self.letter_pool = []
        self.letter_pool_pos = {} # Maps letter index to screen position
        self.program_string = ""
        self.target_program = ""
        
        self.magnetized_letter = None # e.g., {'char': 'A', 'from_pool_idx': 3}
        self.magnetized_letter_pos = pygame.Vector2(0, 0)
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = collections.deque()

        self.last_feedback = {"msg": "", "color": self.COLOR_TEXT, "life": 0}

    def _generate_level(self):
        self.level += 1
        self.program_string = ""
        self.landscape_grid.fill(0) # 0: empty, 1: stable, 2: unstable

        # Difficulty scaling
        prog_len_increase = self.level // 3
        unstable_blocks = (self.level // 5) * 2

        # Define potential puzzles
        puzzles = [
            ("FILL(4,8,4,2)", "A 4x2 chasm needs filling."),
            ("BRIDGE(3,5,12)", "Bridge a gap on row 3."),
            ("CLEAR(2,15,3,3)", "Clear unstable blocks."),
            ("TOWER(6,2,5)", "Build a 5-block high tower."),
        ]
        
        # Select a puzzle
        puzzle_idx = (self.level - 1) % len(puzzles)
        self.target_program, _ = puzzles[puzzle_idx]
        
        # Modify target program based on difficulty
        if prog_len_increase > 0:
            self.target_program += f";RND({random.randint(0,9)})"

        # Create landscape based on puzzle
        if "FILL" in self.target_program or "BRIDGE" in self.target_program:
             # Create a chasm
            for r in range(self.GRID_ROWS // 2 - 1, self.GRID_ROWS // 2 + 1):
                for c in range(self.GRID_COLS):
                    self.landscape_grid[r, c] = 1
            for r in range(self.GRID_ROWS // 2 - 1, self.GRID_ROWS // 2 + 1):
                for c in range(4, self.GRID_COLS-4):
                    self.landscape_grid[r, c] = 0 # The chasm
        else: # Default landscape
            self.landscape_grid[self.GRID_ROWS-2:, :] = 1

        # Add unstable blocks
        for _ in range(unstable_blocks):
            r, c = self.np_random.integers(0, self.GRID_ROWS), self.np_random.integers(0, self.GRID_COLS)
            if self.landscape_grid[r, c] == 1:
                self.landscape_grid[r, c] = 2

        # Populate letter pool
        distractors = "XYZ#?!$@&"
        num_distractors = 5 + self.level // 2
        
        pool_chars = list(self.target_program)
        for _ in range(num_distractors):
            pool_chars.append(random.choice(distractors))
        
        random.shuffle(pool_chars)
        self.letter_pool = pool_chars
        self._calculate_letter_pool_positions()
        
    def _calculate_letter_pool_positions(self):
        self.letter_pool_pos.clear()
        start_x = 20
        y = self.UI_TOP_BAR_H + self.UI_LETTER_POOL_H // 2
        for i, char in enumerate(self.letter_pool):
            x = start_x + i * 25
            self.letter_pool_pos[i] = pygame.Vector2(x, y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.magnetized_letter = None
        self.particles.clear()
        self.last_feedback['life'] = 0

        if options and 'level' in options:
            self.level = options['level'] -1
        else:
            self.level = 0
        
        self._generate_level()
        
        self.cursor_pos = [self.cursor_grid_width // 2, 0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = False

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - self.CURSOR_SPEED)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.cursor_grid_height - 1, self.cursor_pos[1] + self.CURSOR_SPEED)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - self.CURSOR_SPEED)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.cursor_grid_width - 1, self.cursor_pos[0] + self.CURSOR_SPEED)

        # 2. Handle magnetize/release (Spacebar)
        if space_pressed:
            # Sfx: Magnetize/Demagnetize sound
            reward += self._handle_magnetize()

        # 3. Handle teleport/place (Shift)
        if shift_pressed and self.magnetized_letter:
            # Sfx: Teleport/Place sound
            reward += self._handle_place_letter()
        
        self.steps += 1
        if self.last_feedback['life'] > 0:
            self.last_feedback['life'] -= 1

        # --- Termination Check ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not terminated:
             # Check if solution is still possible
            required_chars = collections.Counter(self.target_program[len(self.program_string):])
            available_chars = collections.Counter(self.letter_pool)
            if self.magnetized_letter:
                available_chars[self.magnetized_letter['char']] += 1

            if any(required_chars[char] > available_chars[char] for char in required_chars):
                terminated = True
                self.game_over = True
                reward = -100
                self._show_feedback("IMPOSSIBLE", self.COLOR_FAIL, 60)
                # Sfx: Failure sound
        
        if terminated and not self.game_over: # Max steps reached
            reward = 0 
        
        self.score += reward
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_cursor_context(self):
        """Determines what area the cursor is in."""
        cx, cy = self.cursor_pos
        if cy == 0: # Program Area
            return "program", cx
        if cy == 1: # Letter Pool Area
            return "pool", cx
        return "landscape", (cx, cy - 2)

    def _handle_magnetize(self):
        if self.magnetized_letter: # Release current letter
            self.letter_pool.append(self.magnetized_letter['char'])
            self._calculate_letter_pool_positions()
            self.magnetized_letter = None
            self._create_particles(self._get_cursor_pixel_pos(), self.COLOR_MAGNETIZED)
            return 0

        context, pos = self._get_cursor_context()
        
        if context == "pool":
            # Find closest letter in pool
            cursor_px = self._get_cursor_pixel_pos()
            closest_dist = float('inf')
            closest_idx = -1
            
            # Find the actual index in letter_pool, not the key from letter_pool_pos
            closest_pool_idx = -1
            
            # Recreate a temporary list of (pool_idx, letter_pos) to iterate over
            temp_pool_items = list(self.letter_pool_pos.items())
            
            for i, letter_pos in temp_pool_items:
                dist = cursor_px.distance_to(letter_pos)
                if dist < closest_dist and dist < self.CELL_SIZE:
                    closest_dist = dist
                    closest_idx = i

            if closest_idx != -1:
                # find the actual index in the list
                for pool_idx, char in enumerate(self.letter_pool):
                     # This is a bit fragile if there are duplicate letters, but should work
                     # as letter_pool_pos is rebuilt each time.
                     if self.letter_pool_pos.get(pool_idx) == self.letter_pool_pos.get(closest_idx):
                         closest_pool_idx = pool_idx
                         break
            
            if closest_pool_idx != -1:
                char = self.letter_pool.pop(closest_pool_idx)
                self.magnetized_letter = {'char': char, 'from_pool_idx': closest_idx}
                self.magnetized_letter_pos = self.letter_pool_pos[closest_idx]
                self._calculate_letter_pool_positions()
                self._create_particles(self.magnetized_letter_pos, self.COLOR_MAGNETIZED)
                return 0.1 # Small reward for interaction

        elif context == "program":
            # Pick up from program string
            if pos < len(self.program_string):
                char = self.program_string[pos]
                self.program_string = self.program_string[:pos] + self.program_string[pos+1:]
                self.magnetized_letter = {'char': char, 'from_pool_idx': -1} # -1 indicates from program
                cursor_px = self._get_cursor_pixel_pos()
                self.magnetized_letter_pos = cursor_px
                self._create_particles(cursor_px, self.COLOR_MAGNETIZED)
                return -0.2 # Penalty for deconstructing program
        
        return 0

    def _handle_place_letter(self):
        context, pos = self._get_cursor_context()
        
        if context == "program":
            # Place letter into program string
            char_to_place = self.magnetized_letter['char']
            
            # Insert character at cursor position
            prog_list = list(self.program_string)
            # Clamp insertion position to the end of the string
            insert_pos = min(pos, len(prog_list))
            prog_list.insert(insert_pos, char_to_place)
            new_program = "".join(prog_list)

            self.magnetized_letter = None
            
            # Check if placement is correct
            if self.target_program.startswith(new_program):
                self.program_string = new_program
                self._show_feedback(f"+1 CORRECT: {char_to_place}", self.COLOR_SUCCESS, 30)
                # Sfx: Correct placement sound
                
                # Check for win condition
                if self.program_string == self.target_program:
                    self._execute_program()
                    self.game_over = True
                    self._show_feedback("PUZZLE SOLVED!", self.COLOR_SUCCESS, 120)
                    # Sfx: Level complete fanfare
                    return 100
                return 1.0
            else:
                # Incorrect placement, penalize and reset program
                self.letter_pool.append(char_to_place)
                self._calculate_letter_pool_positions()
                self.program_string = ""
                self._show_feedback(f"-1 MISTAKE!", self.COLOR_FAIL, 30)
                # Sfx: Error sound
                return -1.0
        
        elif context == "pool": # Dropping back in the pool
             self.letter_pool.append(self.magnetized_letter['char'])
             self._calculate_letter_pool_positions()
             self.magnetized_letter = None
             return -0.1 # Small penalty for indecision

        return 0 # No reward if dropped on landscape

    def _execute_program(self):
        """Visually and logically executes the completed program."""
        # This is a simplified placeholder for actual terraforming logic
        # For this example, we'll just turn the chasm green
        if "FILL" in self.target_program or "BRIDGE" in self.target_program:
            for r in range(self.GRID_ROWS // 2 - 1, self.GRID_ROWS // 2 + 1):
                for c in range(4, self.GRID_COLS-4):
                    if self.landscape_grid[r, c] == 0:
                        self.landscape_grid[r, c] = 1
                        pos = (c * self.CELL_SIZE + self.CELL_SIZE / 2, 
                               self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2)
                        self._create_particles(pos, self.COLOR_STABLE, 20)
    
    def _create_particles(self, pos, color, count=15, lifespan=20, speed=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': lifespan + random.randint(-5, 5),
                'color': color
            })

    def _update_and_draw_particles(self):
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                # Draw particle with fade out
                alpha = max(0, min(255, int(255 * (p['life'] / 20))))
                color_with_alpha = p['color'] + (alpha,) if len(p['color']) == 3 else p['color']
                size = int(max(1, 4 * (p['life'] / 20)))
                rect = pygame.Rect(int(p['pos'].x - size/2), int(p['pos'].y - size/2), size, size)
                
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color_with_alpha, temp_surf.get_rect())
                self.screen.blit(temp_surf, rect.topleft)
                i += 1


    def _get_cursor_pixel_pos(self):
        cx, cy = self.cursor_pos
        context, pos = self._get_cursor_context()
        if context == "program":
            px = 20 + pos * 15 # Monospace font assumption
            py = self.UI_PROGRAM_AREA_Y + 20
        elif context == "pool":
            px = 20 + pos * 25
            py = self.UI_LETTER_POOL_Y - 10
        else: # landscape
            l_cx, l_cy = pos
            px = self.GRID_OFFSET_X + l_cx * self.CELL_SIZE + self.CELL_SIZE // 2
            py = self.GRID_OFFSET_Y + l_cy * self.CELL_SIZE + self.CELL_SIZE // 2
        return pygame.Vector2(px, py)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "program": self.program_string,
            "target": self.target_program,
        }

    def _show_feedback(self, msg, color, life):
        self.last_feedback = {"msg": msg, "color": color, "life": life}

    def _render_game(self):
        self._render_landscape()
        self._render_ui()
        self._update_and_draw_particles()
        self._render_cursor()
        self._render_magnetized_letter()

    def _render_landscape(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                state = self.landscape_grid[r, c]
                color = self.COLOR_EMPTY
                edge_color = self.COLOR_GRID
                if state == 1:
                    color = self.COLOR_STABLE
                    edge_color = self.COLOR_STABLE_EDGE
                elif state == 2:
                    color = self.COLOR_UNSTABLE
                    edge_color = self.COLOR_UNSTABLE_EDGE

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, edge_color, rect, 2)
    
    def _render_ui(self):
        # Panel backgrounds
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.SCREEN_WIDTH, self.UI_TOP_BAR_H))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, self.UI_TOP_BAR_H, self.SCREEN_WIDTH, self.UI_LETTER_POOL_H))
        
        # Score and Level text
        score_surf = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        level_surf = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        self.screen.blit(level_surf, (10, 10))

        # Program String
        program_label = self.font_code.render("PROG:", True, self.COLOR_TEXT)
        self.screen.blit(program_label, (10, self.UI_PROGRAM_AREA_Y + 10))
        
        program_full_text = self.program_string + "_" * (len(self.target_program) - len(self.program_string))
        program_surf = self.font_code.render(program_full_text, True, self.COLOR_TEXT)
        self.screen.blit(program_surf, (program_label.get_width() + 20, self.UI_PROGRAM_AREA_Y + 10))

        # Letter Pool
        pool_label = self.font_code.render("POOL:", True, self.COLOR_TEXT)
        self.screen.blit(pool_label, (10, self.UI_LETTER_POOL_Y - 20))
        for i, char in enumerate(self.letter_pool):
            # The key in letter_pool_pos corresponds to the original index, not the current one.
            # We need to find the correct key for the current letter `char` at index `i`.
            # This logic is tricky. Instead, let's just use the calculated positions.
            # Re-calculating positions every frame is inefficient but safe.
            x = 20 + i * 25
            y = self.UI_TOP_BAR_H + self.UI_LETTER_POOL_H // 2
            pos = pygame.Vector2(x, y)
            
            # Store the calculated position for magnetize logic to use
            self.letter_pool_pos[i] = pos

            char_surf = self.font_code.render(char, True, self.COLOR_TEXT)
            char_rect = char_surf.get_rect(center=pos)
            self.screen.blit(char_surf, char_rect)


        # Feedback message
        if self.last_feedback['life'] > 0:
            alpha = max(0, min(255, self.last_feedback['life'] * 5))
            feedback_surf = self.font_main.render(self.last_feedback['msg'], True, self.last_feedback['color'])
            feedback_surf.set_alpha(alpha)
            self.screen.blit(feedback_surf, feedback_surf.get_rect(centerx=self.SCREEN_WIDTH/2, y=self.SCREEN_HEIGHT-35))

    def _render_cursor(self):
        pos = self._get_cursor_pixel_pos()
        size = self.CELL_SIZE // 4 # Made cursor smaller
        
        # Pulsing glow effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # Varies between 0 and 1
        glow_size = int(size * (2.5 + pulse * 1.5))
        
        # Draw glow using a surface for alpha blending
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_CURSOR_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pos.x - glow_size, pos.y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (int(pos.x), int(pos.y)), size)
        
    def _render_magnetized_letter(self):
        if self.magnetized_letter:
            target_pos = self._get_cursor_pixel_pos()
            # Interpolate for smooth following
            self.magnetized_letter_pos.x = self.magnetized_letter_pos.x * 0.7 + target_pos.x * 0.3
            self.magnetized_letter_pos.y = self.magnetized_letter_pos.y * 0.7 + target_pos.y * 0.3

            char_surf = self.font_code.render(self.magnetized_letter['char'], True, self.COLOR_MAGNETIZED)
            char_rect = char_surf.get_rect(center=self.magnetized_letter_pos)
            
            # Glow effect
            glow_surf = self.font_code.render(self.magnetized_letter['char'], True, self.COLOR_MAGNETIZED)
            glow_surf.set_alpha(80)
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx != 0 or dy != 0:
                        self.screen.blit(glow_surf, char_rect.move(dx, dy))

            self.screen.blit(char_surf, char_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Ensure the dummy driver is NOT set for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Code Terraformer")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0 # 0=released, 1=held
        shift_held = 0 # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        if keys[pygame.K_r]: # Manual reset
            obs, info = env.reset()
            terminated = False
            
        if not terminated:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # The observation is already the rendered screen, so we just need to display it.
        # Need to transpose it back for pygame's display format.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()