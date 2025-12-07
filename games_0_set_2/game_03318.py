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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the crystal. Match the crystal's color to a target space."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Navigate the crystal to fill all target spaces with matching colors before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        self.grid_extents = (4, 4) # Logical grid size
        self.max_moves = 50

        # --- Color Palette ---
        self.COLORS = {
            "BG": (20, 30, 40),
            "GRID_FLOOR": (45, 55, 70),
            "RED": (255, 80, 80),
            "GREEN": (80, 255, 80),
            "BLUE": (80, 150, 255),
            "GOLD": (255, 215, 0),
            "WHITE": (240, 240, 240),
            "UI_TEXT": (220, 220, 230),
        }
        self.CRYSTAL_COLORS = [self.COLORS["RED"], self.COLORS["GREEN"], self.COLORS["BLUE"]]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State (initialized in reset) ---
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.board_spaces = []
        self.player_pos = (0, 0)
        self.player_color = self.COLORS["WHITE"]
        self.particles = []

        # self.reset() is called by the test harness, no need to call it here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.max_moves
        self.game_over = False
        self.win = False
        self.steps = 0
        self.particles = []

        self._generate_board()
        self._spawn_player()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        reward = 0
        terminated = False

        if movement == 0: # No-op
            # Small penalty for inaction
            reward = -0.1
        else:
            self.steps += 1
            self.moves_left -= 1
            reward = -0.2 # Base cost for making a move

            start_pos = self.player_pos
            target_pos = list(start_pos)
            
            # 1=up, 2=down, 3=left, 4=right
            if movement == 1: target_pos[1] -= 1
            elif movement == 2: target_pos[1] += 1
            elif movement == 3: target_pos[0] -= 1
            elif movement == 4: target_pos[0] += 1
            target_pos = tuple(target_pos)

            if not (0 <= target_pos[0] < self.grid_extents[0] and 0 <= target_pos[1] < self.grid_extents[1]):
                # Penalty for bumping into a wall
                reward = -0.5
                # sfx: bump
            else:
                self.player_pos = target_pos
                match_found = False
                for space in self.board_spaces:
                    # FIX: Use np.array_equal for element-wise comparison of colors.
                    # The original comparison `==` on a numpy array (player_color) and a tuple (space['color'])
                    # results in a boolean array, which cannot be used in an `if` statement.
                    if space['pos'] == self.player_pos and not space['filled'] and np.array_equal(space['color'], self.player_color):
                        space['filled'] = True
                        self.score += 1
                        reward = 1.0
                        match_found = True
                        self._create_particles(self.player_pos, self.COLORS['GOLD'])
                        # sfx: match_success
                        break
                
                if match_found:
                    self._spawn_player()

        # Check for game termination
        if all(s['filled'] for s in self.board_spaces):
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
            # sfx: win_fanfare
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward += -10
            # sfx: lose_sound
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLORS["BG"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _generate_board(self):
        self.board_layout = [(1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)]
        colors = self.CRYSTAL_COLORS * 4 
        self.np_random.shuffle(colors)
        self.board_spaces = []
        for i, pos in enumerate(self.board_layout):
            self.board_spaces.append({
                'pos': pos,
                'color': colors[i],
                'filled': False
            })

    def _spawn_player(self):
        board_pos_set = {s['pos'] for s in self.board_spaces}
        valid_spawns = []
        for r in range(self.grid_extents[0]):
            for c in range(self.grid_extents[1]):
                if (r, c) not in board_pos_set:
                    valid_spawns.append((r, c))
        
        if not valid_spawns: valid_spawns = [(0,0)] # Failsafe

        self.player_pos = tuple(self.np_random.choice(valid_spawns, axis=0))
        self.player_color = self.np_random.choice(self.CRYSTAL_COLORS, axis=0)

    def _iso_to_screen(self, r, c):
        origin_x = self.screen_width / 2
        origin_y = self.screen_height / 2 - 60
        tile_width_half = 32
        tile_height_half = 16
        screen_x = origin_x + (r - c) * tile_width_half
        screen_y = origin_y + (r + c) * tile_height_half
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, pos, color, is_player=False):
        r, c = pos
        sx, sy = self._iso_to_screen(r, c)
        
        tile_width_half = 32
        tile_height_half = 16
        depth = 12

        points = [
            (sx, sy - tile_height_half),
            (sx + tile_width_half, sy),
            (sx, sy + tile_height_half),
            (sx - tile_width_half, sy)
        ]

        if is_player:
            # Draw player crystal slightly elevated and smaller
            offset = 8
            player_points = [
                (sx, sy - tile_height_half - offset),
                (sx + tile_width_half, sy - offset),
                (sx, sy + tile_height_half - offset),
                (sx - tile_width_half, sy - offset)
            ]
            
            dark_color = tuple(max(0, val - 60) for val in color)
            highlight_color = tuple(min(255, val + 60) for val in color)
            
            # Shadow
            shadow_points = [(p[0], p[1] + offset + 2) for p in player_points]
            pygame.gfxdraw.filled_polygon(self.screen, shadow_points, (0,0,0,50))

            # Body
            pygame.gfxdraw.filled_polygon(self.screen, player_points, color)
            pygame.gfxdraw.aapolygon(self.screen, player_points, dark_color)
            
            # Highlight
            pygame.draw.line(self.screen, highlight_color, player_points[0], player_points[3], 2)
            pygame.draw.line(self.screen, highlight_color, player_points[0], player_points[1], 2)

        else:
            # Draw the 3D block
            darker_color = tuple(max(0, val - 40) for val in color)
            side_color = tuple(max(0, val - 20) for val in color)
            
            # Top face
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Side faces
            left_face = [points[2], points[3], (points[3][0], points[3][1] + depth), (points[2][0], points[2][1] + depth)]
            right_face = [points[1], points[2], (points[2][0], points[2][1] + depth), (points[1][0], points[1][1] + depth)]
            pygame.gfxdraw.filled_polygon(self.screen, left_face, side_color)
            pygame.gfxdraw.filled_polygon(self.screen, right_face, darker_color)

            # Outline
            pygame.gfxdraw.aapolygon(self.screen, points, darker_color)

    def _render_game(self):
        # Render floor tiles
        for r in range(self.grid_extents[0]):
            for c in range(self.grid_extents[1]):
                self._draw_iso_tile((r, c), self.COLORS['GRID_FLOOR'])

        # Render target spaces
        for space in self.board_spaces:
            color = self.COLORS['GOLD'] if space['filled'] else space['color']
            self._draw_iso_tile(space['pos'], color)

        # Render particles
        self._update_particles()
        for p in self.particles:
            alpha_color = (*p['color'], p['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), alpha_color)

        # Render player
        if not self.game_over or (self.game_over and not self.win):
             self._draw_iso_tile(self.player_pos, self.player_color, is_player=True)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLORS["UI_TEXT"])
        self.screen.blit(score_text, (10, 10))

        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLORS["UI_TEXT"])
        self.screen.blit(moves_text, (self.screen_width - moves_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLORS["GOLD"] if self.win else self.COLORS["RED"]
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, grid_pos, color):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(20, 50)
            self.particles.append({
                'pos': [sx, sy],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.np_random.uniform(2, 6),
                'alpha': 255
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                p['alpha'] = int(255 * (p['life'] / p['max_life']))
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    # env.validate_implementation() # This would fail before reset is called
    obs, info = env.reset(seed=123)
    
    # --- Pygame loop for human play ---
    # To run this, you need to unset the dummy video driver.
    # E.g., run `unset SDL_VIDEODRIVER` in your shell before executing the script.
    # Or comment out the `os.environ.setdefault` line at the top.
    try:
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Crystal Cavern")
        running = True
    except pygame.error as e:
        print(f"Could not set up display for human play: {e}")
        print("This is expected if you are running in a headless environment.")
        print("The environment is still valid for training.")
        running = False
        
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op, no buttons

    if running:
        print("\n" + "="*30)
        print("Human Play Mode")
        print(env.user_guide)
        print("="*30 + "\n")

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                
                # Set action based on key press
                move = 0
                if event.key == pygame.K_UP: move = 1
                elif event.key == pygame.K_DOWN: move = 2
                elif event.key == pygame.K_LEFT: move = 3
                elif event.key == pygame.K_RIGHT: move = 4
                
                if move != 0:
                    action = np.array([move, 0, 0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")
                    if terminated:
                        print("--- Episode Finished ---")

        # --- Drawing ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()