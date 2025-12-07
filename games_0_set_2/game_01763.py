import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press Space to swap the selected block "
        "in the direction you last moved. Press Shift to shuffle the board (costs 3 moves)."
    )

    game_description = (
        "A colorful match-3 puzzle game. Swap blocks to create lines of 3 or more. "
        "Clear the board before you run out of moves! Plan your swaps carefully to create cascading combos."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game Constants ---
        self.GRID_DIM = 8
        self.BLOCK_SIZE = 40
        self.NUM_BLOCK_TYPES = 5
        self.GRID_WIDTH = self.GRID_DIM * self.BLOCK_SIZE
        self.GRID_HEIGHT = self.GRID_DIM * self.BLOCK_SIZE
        self.GRID_X_OFFSET = (self.screen_width - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.screen_height - self.GRID_HEIGHT) // 2
        self.ANIMATION_SPEED = 0.2  # Progress per frame

        # --- Colors ---
        self.COLOR_BG = (40, 40, 50)
        self.COLOR_GRID = (60, 60, 70)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.last_action = None
        self.steps = None
        self.score = None
        self.moves_remaining = None
        self.game_over = None
        self.game_phase = None
        self.animation_progress = None
        self.animation_data = None
        self.particles = None
        self.turn_reward = None
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = 20
        self.game_over = False
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.last_move_dir = (0, 0)
        self.last_action = np.array([0, 0, 0])
        self.particles = []
        self.turn_reward = 0
        self._generate_initial_board()
        self.game_phase = "IDLE"
        self.animation_progress = 0
        self.animation_data = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.turn_reward = 0

        self._handle_input(action)
        self._update_game_state()
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.moves_remaining <= 0 and np.any(self.grid > 0):
                self.turn_reward += -50 # Loss
            elif np.all(self.grid == 0):
                self.turn_reward += 100 # Win
            self.game_phase = "GAME_OVER"


        return (
            self._get_observation(),
            self.turn_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_phase != "IDLE":
            return # Ignore input during animations

        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and self.last_action[1] == 0
        shift_pressed = shift_action == 1 and self.last_action[2] == 0
        self.last_action = action

        # Cursor movement
        moved = False
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            self.last_move_dir = (0, -1)
            moved = True
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1)
            self.last_move_dir = (0, 1)
            moved = True
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            self.last_move_dir = (-1, 0)
            moved = True
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1)
            self.last_move_dir = (1, 0)
            moved = True
        
        # Swap action
        if space_pressed and self.last_move_dir != (0, 0) and self.moves_remaining > 0:
            x1, y1 = self.cursor_pos
            x2, y2 = x1 + self.last_move_dir[0], y1 + self.last_move_dir[1]

            if 0 <= x2 < self.GRID_DIM and 0 <= y2 < self.GRID_DIM:
                self.moves_remaining -= 1
                self.game_phase = "SWAPPING"
                self.animation_progress = 0
                self.animation_data = {"pos1": (x1, y1), "pos2": (x2, y2)}

        # Shuffle action
        if shift_pressed and self.moves_remaining >= 3:
            self.moves_remaining -= 3
            self.game_phase = "SHUFFLING"
            self.animation_progress = 0
            self.animation_data = {}
    
    def _update_game_state(self):
        if self.animation_progress < 1.0:
            self.animation_progress = min(1.0, self.animation_progress + self.ANIMATION_SPEED)

        if self.animation_progress < 1.0:
            return

        # --- State Machine Transitions ---
        if self.game_phase == "SWAPPING":
            p1, p2 = self.animation_data["pos1"], self.animation_data["pos2"]
            val1 = self.grid[p1[1], p1[0]]
            self.grid[p1[1], p1[0]] = self.grid[p2[1], p2[0]]
            self.grid[p2[1], p2[0]] = val1
            self.game_phase = "CHECKING"
            self.animation_data = {"is_swap": True} # Flag to track if this check came from a swap

        elif self.game_phase == "SHUFFLING":
            flat_grid = self.grid.flatten()
            self.rng.shuffle(flat_grid)
            self.grid = flat_grid.reshape((self.GRID_DIM, self.GRID_DIM))
            self.game_phase = "CHECKING"
            self.animation_data = {"is_swap": False}

        elif self.game_phase == "CHECKING":
            matches = self._find_matches()
            if matches:
                self._process_matches(matches)
                self.game_phase = "CLEARING"
                self.animation_progress = 0
                self.animation_data = {"matches": matches}
            else:
                if self.animation_data.get("is_swap", False):
                    self.turn_reward = -0.2 # Penalty for non-productive move
                self.game_phase = "IDLE"
                self.animation_data = {}

        elif self.game_phase == "CLEARING":
            matches = self.animation_data["matches"]
            for r, c in matches:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0 # 0 means empty
            self.game_phase = "FALLING"
            self.animation_progress = 0
            self.animation_data = self._calculate_fall_data()

        elif self.game_phase == "FALLING":
            # Apply the fall instantly in the grid data
            new_grid = np.zeros_like(self.grid)
            for c in range(self.GRID_DIM):
                write_idx = self.GRID_DIM - 1
                for r in range(self.GRID_DIM - 1, -1, -1):
                    if self.grid[r, c] != 0:
                        new_grid[write_idx, c] = self.grid[r, c]
                        write_idx -= 1
            self.grid = new_grid
            
            # Refill
            self.grid[self.grid == 0] = self.rng.integers(1, self.NUM_BLOCK_TYPES + 1, size=np.sum(self.grid == 0))
            
            self.game_phase = "CHECKING"
            self.animation_data = {"is_swap": False} # It's a cascade, not a user swap

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
            "moves_remaining": self.moves_remaining,
            "game_phase": self.game_phase,
        }
        
    def _calculate_reward(self):
        return self.turn_reward

    def _check_termination(self):
        # The result of np.all() can be a numpy.bool_, which is not an instance of Python's bool.
        # We explicitly cast it to a standard bool to satisfy the Gymnasium API.
        is_terminated = self.moves_remaining <= 0 or np.all(self.grid == 0)
        return bool(is_terminated)

    # --- Generation and Logic ---

    def _generate_initial_board(self):
        self.grid = self.rng.integers(1, self.NUM_BLOCK_TYPES + 1, size=(self.GRID_DIM, self.GRID_DIM))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.rng.integers(1, self.NUM_BLOCK_TYPES + 1)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_DIM):
            for r in range(self.GRID_DIM - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _process_matches(self, matches):
        num_cleared = len(matches)
        self.score += num_cleared
        self.turn_reward += num_cleared
        if num_cleared >= 4:
            bonus = 5
            self.score += bonus
            self.turn_reward += bonus
    
    def _calculate_fall_data(self):
        fall_data = []
        for c in range(self.GRID_DIM):
            empty_count = 0
            for r in range(self.GRID_DIM - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    fall_data.append({
                        "from": (c, r), "to": (c, r + empty_count), "type": self.grid[r, c]
                    })
        return {"falls": fall_data}
        
    # --- Rendering ---

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            x = self.GRID_X_OFFSET + i * self.BLOCK_SIZE
            y = self.GRID_Y_OFFSET + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))

        # Draw blocks
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                block_type = self.grid[r, c]
                if block_type == 0:
                    continue

                color = self.BLOCK_COLORS[block_type - 1]
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.BLOCK_SIZE,
                    self.GRID_Y_OFFSET + r * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                
                # Default render position
                render_rect = rect.copy()
                
                # --- Handle Animations ---
                t = self.animation_progress
                if self.game_phase == "SWAPPING":
                    p1, p2 = self.animation_data["pos1"], self.animation_data["pos2"]
                    if (c, r) == p1:
                        render_rect.x = int(self._lerp(rect.x, self.GRID_X_OFFSET + p2[0] * self.BLOCK_SIZE, t))
                        render_rect.y = int(self._lerp(rect.y, self.GRID_Y_OFFSET + p2[1] * self.BLOCK_SIZE, t))
                    elif (c, r) == p2:
                        render_rect.x = int(self._lerp(rect.x, self.GRID_X_OFFSET + p1[0] * self.BLOCK_SIZE, t))
                        render_rect.y = int(self._lerp(rect.y, self.GRID_Y_OFFSET + p1[1] * self.BLOCK_SIZE, t))

                elif self.game_phase == "CLEARING":
                    if (r, c) in self.animation_data["matches"]:
                        scale = self._lerp(1, 0, t)
                        render_rect.width = int(self.BLOCK_SIZE * scale)
                        render_rect.height = int(self.BLOCK_SIZE * scale)
                        render_rect.center = rect.center
                
                elif self.game_phase == "FALLING":
                    is_falling = False
                    for fall in self.animation_data["falls"]:
                        if (c, r) == fall["from"]:
                            start_y = self.GRID_Y_OFFSET + fall["from"][1] * self.BLOCK_SIZE
                            end_y = self.GRID_Y_OFFSET + fall["to"][1] * self.BLOCK_SIZE
                            render_rect.y = int(self._lerp(start_y, end_y, t))
                            is_falling = True
                            break
                    if not is_falling and self.grid[r,c] != 0: # Draw non-falling blocks
                        pygame.draw.rect(self.screen, color, rect, border_radius=5)
                        pygame.draw.rect(self.screen, tuple(min(255, x+30) for x in color), rect.inflate(-10,-10), border_radius=5)

                if self.game_phase != "FALLING" or is_falling:
                    pygame.draw.rect(self.screen, color, render_rect, border_radius=5)
                    pygame.draw.rect(self.screen, tuple(min(255, x+30) for x in color), render_rect.inflate(-10,-10), border_radius=5)

        # Draw cursor
        if self.game_phase == "IDLE":
            cursor_rect = pygame.Rect(
                self.GRID_X_OFFSET + self.cursor_pos[0] * self.BLOCK_SIZE,
                self.GRID_Y_OFFSET + self.cursor_pos[1] * self.BLOCK_SIZE,
                self.BLOCK_SIZE, self.BLOCK_SIZE
            )
            glow_t = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1 pulse
            glow_alpha = int(self._lerp(100, 200, glow_t))
            pygame.gfxdraw.rectangle(self.screen, cursor_rect.inflate(4,4), (*self.COLOR_CURSOR, glow_alpha))
            pygame.gfxdraw.rectangle(self.screen, cursor_rect.inflate(3,3), (*self.COLOR_CURSOR, glow_alpha))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_ui(self):
        # Score and Moves
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, 10))
        
        # Game Over Message
        if self.game_phase == "GAME_OVER":
            if np.all(self.grid == 0):
                msg = "YOU WIN!"
                color = (150, 255, 150)
            else:
                msg = "GAME OVER"
                color = (255, 150, 150)
            
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)


    # --- Particles & Effects ---

    def _create_particles(self, c, r, block_type):
        px = self.GRID_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[block_type - 1]
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': self.rng.random() * 3 + 2,
                'lifetime': self.rng.integers(20, 40),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['lifetime'] -= 1
            p['size'] -= 0.05
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0]
        
    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    running = True
    terminated = False
    
    action = np.array([0, 0, 0]) # No-op, no space, no shift
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        mov_action = 0 # No-op
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Create a new action only if a key is pressed to avoid rapid-fire actions
        current_action = np.array([mov_action, space_action, shift_action])

        # --- Gym Step ---
        if not terminated:
            # The environment handles action logic, including debouncing via last_action
            obs, reward, terminated, truncated, info = env.step(current_action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
        
        if terminated:
            # Keep rendering the final state without taking actions
            obs = env.render()
        
        # --- Render to screen ---
        # The observation is a numpy array, but we can blit the env's internal surface
        # which is more direct for human play.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    pygame.quit()