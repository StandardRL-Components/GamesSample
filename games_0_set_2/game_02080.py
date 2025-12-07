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


def draw_rounded_rect(surface, rect, color, corner_radius):
    """
    Draw a filled rectangle with rounded corners and anti-aliasing.
    This is a substitute for the non-existent pygame.gfxdraw.rounded_rectangle.
    """
    if rect.width < 2 * corner_radius or rect.height < 2 * corner_radius:
        # If the rectangle is too small for the radius, draw a simple rectangle.
        if rect.width < 1 or rect.height < 1:
            return
        pygame.draw.rect(surface, color, rect)
        return

    # Central rectangles
    pygame.draw.rect(surface, color, (rect.left + corner_radius, rect.top, rect.width - 2 * corner_radius, rect.height))
    pygame.draw.rect(surface, color, (rect.left, rect.top + corner_radius, rect.width, rect.height - 2 * corner_radius))

    # Rounded corners
    # filled_circle and aacircle are used to create anti-aliased filled circles
    pygame.gfxdraw.filled_circle(surface, rect.left + corner_radius, rect.top + corner_radius, corner_radius, color)
    pygame.gfxdraw.aacircle(surface, rect.left + corner_radius, rect.top + corner_radius, corner_radius, color)

    pygame.gfxdraw.filled_circle(surface, rect.right - corner_radius - 1, rect.top + corner_radius, corner_radius, color)
    pygame.gfxdraw.aacircle(surface, rect.right - corner_radius - 1, rect.top + corner_radius, corner_radius, color)

    pygame.gfxdraw.filled_circle(surface, rect.left + corner_radius, rect.bottom - corner_radius - 1, corner_radius, color)
    pygame.gfxdraw.aacircle(surface, rect.left + corner_radius, rect.bottom - corner_radius - 1, corner_radius, color)

    pygame.gfxdraw.filled_circle(surface, rect.right - corner_radius - 1, rect.bottom - corner_radius - 1, corner_radius, color)
    pygame.gfxdraw.aacircle(surface, rect.right - corner_radius - 1, rect.bottom - corner_radius - 1, corner_radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the selector. Space to grab or drop a colored block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Sort the colored blocks into their matching zones before the time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (45, 48, 52)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SELECTOR_BRIGHT = (255, 255, 0)
    COLOR_SELECTOR_DARK = (200, 200, 0)
    BLOCK_COLORS = [
        (255, 87, 87),   # Red
        (87, 255, 150),  # Green
        (87, 150, 255),  # Blue
        (255, 230, 87),  # Yellow
    ]

    # Grid settings
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    
    NUM_BLOCKS_PER_COLOR = 3
    NUM_COLORS = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_icon = pygame.font.SysFont("Arial", 24, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        self.selector_pos = [0, 0]
        self.blocks = []
        self.target_zones = {}
        self.held_block_idx = None
        self.prev_space_held = False
        self.particles = []
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2 - 2]
        self.held_block_idx = None
        self.prev_space_held = False
        self.particles = []

        self._initialize_blocks_and_zones()

        return self._get_observation(), self._get_info()

    def _initialize_blocks_and_zones(self):
        self.blocks = []
        
        # Define target zones at the bottom
        zone_height = 2
        zone_width = self.GRID_COLS // self.NUM_COLORS
        for i in range(self.NUM_COLORS):
            self.target_zones[i] = pygame.Rect(
                i * zone_width, 
                self.GRID_ROWS - zone_height, 
                zone_width, 
                zone_height
            )

        # Generate blocks in the upper area
        available_positions = []
        play_area_rows = self.GRID_ROWS - zone_height - 1
        for r in range(play_area_rows):
            for c in range(self.GRID_COLS):
                available_positions.append([c, r])
        
        self.np_random.shuffle(available_positions)

        block_id = 0
        for color_idx in range(self.NUM_COLORS):
            for _ in range(self.NUM_BLOCKS_PER_COLOR):
                if not available_positions:
                    raise RuntimeError("Not enough space to place all blocks.")
                pos = available_positions.pop()
                self.blocks.append({
                    "id": block_id,
                    "color_idx": color_idx,
                    "pos": pos,
                    "is_locked": False,
                    "grab_pos": None,
                })
                block_id += 1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief

        # Handle movement
        if movement == 1: self.selector_pos[1] -= 1  # Up
        elif movement == 2: self.selector_pos[1] += 1  # Down
        elif movement == 3: self.selector_pos[0] -= 1  # Left
        elif movement == 4: self.selector_pos[0] += 1  # Right
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_COLS - 1)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_ROWS - 1)
        
        # Handle interaction (grab/drop on space press)
        if space_held and not self.prev_space_held:
            reward += self._handle_interaction()
        self.prev_space_held = space_held

        # Update game logic
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.timer <= 0:
                reward -= 100.0 # Timeout penalty
            else: # Victory
                reward += 100.0 # Win bonus
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_interaction(self):
        # --- DROP LOGIC ---
        if self.held_block_idx is not None:
            block = self.blocks[self.held_block_idx]
            
            # Check if target cell is occupied
            is_occupied = any(
                b["pos"] == self.selector_pos and i != self.held_block_idx
                for i, b in enumerate(self.blocks)
            )

            if not is_occupied:
                # Drop the block
                block["pos"] = list(self.selector_pos) # Make a copy
                reward = self._calculate_drop_reward(block)
                
                # Check for correct placement
                target_zone = self.target_zones[block["color_idx"]]
                if target_zone.collidepoint(self.selector_pos):
                    block["is_locked"] = True
                    reward += 1.0 # Correct placement bonus
                    # SFX: Correct place sound
                    self._create_particles(self._grid_to_screen(self.selector_pos), self.BLOCK_COLORS[block["color_idx"]])
                else:
                    # SFX: Block drop sound
                    pass

                self.held_block_idx = None
                return reward
            else:
                # SFX: Cannot drop sound
                return 0.0 # Cannot drop on occupied space

        # --- GRAB LOGIC ---
        else:
            for i, block in enumerate(self.blocks):
                if not block["is_locked"] and block["pos"] == self.selector_pos:
                    self.held_block_idx = i
                    block["grab_pos"] = list(self.selector_pos)
                    # SFX: Grab sound
                    return 0.0 # No reward for grabbing
        return 0.0

    def _calculate_drop_reward(self, block):
        if block["grab_pos"] is None:
            return 0.0
        
        target_center = self.target_zones[block["color_idx"]].center
        
        # Manhattan distance
        dist_before = abs(block["grab_pos"][0] - target_center[0]) + abs(block["grab_pos"][1] - target_center[1])
        dist_after = abs(block["pos"][0] - target_center[0]) + abs(block["pos"][1] - target_center[1])
        
        block["grab_pos"] = None # Clear grab position
        
        # Reward for moving closer to the target zone
        return (dist_before - dist_after) * 0.1

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if all(b["is_locked"] for b in self.blocks):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_grid()
        self._render_target_zones()
        self._render_blocks()
        self._render_selector()
        self._render_particles()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "blocks_locked": sum(1 for b in self.blocks if b["is_locked"]),
        }

    # --- Rendering Methods ---
    def _grid_to_screen(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE
        y = grid_pos[1] * self.CELL_SIZE
        return x, y
    
    def _render_grid(self):
        for x in range(0, self.GRID_WIDTH + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GRID_HEIGHT))
        for y in range(0, self.GRID_HEIGHT + 1, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.GRID_WIDTH, y))

    def _render_target_zones(self):
        for color_idx, rect in self.target_zones.items():
            screen_rect = pygame.Rect(
                rect.x * self.CELL_SIZE,
                rect.y * self.CELL_SIZE,
                rect.width * self.CELL_SIZE,
                rect.height * self.CELL_SIZE,
            )
            # Desaturate color for background zone
            zone_color = pygame.Color(self.BLOCK_COLORS[color_idx])
            zone_color.hsva = (zone_color.hsva[0], 50, 50, 100)
            pygame.draw.rect(self.screen, zone_color, screen_rect)

    def _render_blocks(self):
        for i, block in enumerate(self.blocks):
            color = self.BLOCK_COLORS[block["color_idx"]]
            
            if self.held_block_idx == i:
                # Held block follows selector smoothly
                sx, sy = self._grid_to_screen(self.selector_pos)
                rect = pygame.Rect(sx, sy, self.CELL_SIZE, self.CELL_SIZE)
                # Visual feedback for held block
                pygame.draw.rect(self.screen, color, rect.inflate(4, 4), border_radius=8)
                pygame.draw.rect(self.screen, (255,255,255), rect.inflate(4, 4), width=2, border_radius=8)
            else:
                sx, sy = self._grid_to_screen(block["pos"])
                rect = pygame.Rect(sx, sy, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect, border_radius=6)

            if block["is_locked"]:
                # Draw a checkmark/lock icon
                check_text = self.font_icon.render("✓", True, (255, 255, 255))
                text_rect = check_text.get_rect(center=rect.center)
                self.screen.blit(check_text, text_rect)

    def _render_selector(self):
        sx, sy = self._grid_to_screen(self.selector_pos)
        rect = pygame.Rect(sx, sy, self.CELL_SIZE, self.CELL_SIZE)
        
        # Animated glow effect
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = self.COLOR_SELECTOR_BRIGHT
        
        # Use the helper function for anti-aliased rounded rectangle
        draw_rounded_rect(self.screen, rect, (*color, int(alpha)), 6)
        draw_rounded_rect(self.screen, rect.inflate(-2, -2), (*color, int(alpha//2)), 5)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] -= 0.2
        self.particles = [p for p in self.particles if p['radius'] > 0]

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer display
        timer_color = (255, 100, 100) if self.timer < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_large.render(f"TIME: {max(0, self.timer):.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(b["is_locked"] for b in self.blocks)
            msg = "LEVEL CLEAR!" if win_condition else "TIME'S UP!"
            msg_color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, msg_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            score_end_text = self.font_small.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
            score_end_rect = score_end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(score_end_text, score_end_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, ensure you have pygame installed: pip install pygame
    # It will open a window and you can play with arrow keys and space.
    # Un-comment the next line to run the game with a visible window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Sorter")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    # Main game loop
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()