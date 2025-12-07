import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:43:21.665216
# Source Brief: brief_02301.md
# Brief Index: 2301
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    TerraHex is a turn-based strategy game where two players compete to
    terraform a hex-grid world. Each turn, players gain 'Time Points'
    which they can spend to either terraform barren land into fertile territory
    or research technology for powerful bonuses. The player who controls the
    most fertile land after 200 turns wins.

    The environment is designed for a reinforcement learning agent, using a
    MultiDiscrete action space to control a cursor, terraform tiles, and
    research technology. Visuals are a primary focus, with a clean, geometric
    aesthetic and clear feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A turn-based strategy game where two players compete to terraform a hex-grid world by spending Time Points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to terraform a tile or shift to research technology."
    )
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    HEX_RADIUS = 15
    GRID_WIDTH = 17  # Number of hexes horizontally
    GRID_HEIGHT = 11 # Number of hexes vertically

    # Game Rules
    MAX_TURNS = 200
    TECH_LEVELS = 3
    TERRAFORM_COST = 3
    TECH_COSTS = [5, 8, 12] # Cost for tech levels 1, 2, 3

    # Player IDs
    PLAYER_1 = 1
    PLAYER_2 = 2

    # Tile Types
    TILE_BARREN = 0
    TILE_FERTILE = 1
    TILE_WATER = 2
    TILE_MOUNTAIN = 3

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_UI_BG = (40, 50, 70, 200)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    
    PLAYER_COLORS = {
        PLAYER_1: (255, 215, 0), # Gold
        PLAYER_2: (192, 192, 192) # Silver
    }
    TILE_TYPE_COLORS = {
        TILE_BARREN: (139, 119, 101),  # Brown
        TILE_FERTILE: (60, 179, 113),  # Green
        TILE_WATER: (70, 130, 180),    # Blue
        TILE_MOUNTAIN: (105, 105, 105) # Grey
    }

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
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = {}
        self.cursor_pos = (0, 0)
        self.p1_time_points = 0
        self.p1_tech_level = 0
        self.p2_time_points = 0
        self.p2_tech_level = 0
        self.message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.p1_time_points = 5
        self.p1_tech_level = 0
        self.p2_time_points = 5
        self.p2_tech_level = 0
        
        # Grid and cursor
        self._generate_grid()
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self.message = "New game started. Good luck!"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        message_log = []

        # --- Player 1 (Agent) Turn ---
        # 1. Roll for Time Points
        p1_roll = self.np_random.integers(1, 7)
        self.p1_time_points += p1_roll
        # SFX: Dice roll sound
        message_log.append(f"P1 rolled {p1_roll}")

        # 2. Process Action
        self._move_cursor(movement)
        
        if shift_pressed:
            # Research Action
            if self.p1_tech_level < self.TECH_LEVELS:
                cost = self.TECH_COSTS[self.p1_tech_level]
                if self.p1_time_points >= cost:
                    self.p1_time_points -= cost
                    self.p1_tech_level += 1
                    reward += 5
                    self.score += 5
                    message_log.append("P1 researched tech!")
                    # SFX: Tech upgrade success
                else:
                    message_log.append("P1: Not enough TP for tech.")
                    # SFX: Action failed
            else:
                message_log.append("P1: Max tech reached.")
        
        elif space_pressed:
            # Terraform Action
            tile = self.grid.get(self.cursor_pos)
            if tile and tile['type'] == self.TILE_BARREN:
                cost = self.TERRAFORM_COST
                # Tech bonus: Level 1 reduces cost
                if self.p1_tech_level >= 1:
                    cost = max(1, cost - 1)
                
                if self.p1_time_points >= cost:
                    self.p1_time_points -= cost
                    tile['type'] = self.TILE_FERTILE
                    tile['owner'] = self.PLAYER_1
                    reward += 1
                    self.score += 1
                    message_log.append("P1 terraformed a tile!")
                    # SFX: Terraforming success
                else:
                    message_log.append("P1: Not enough TP to terraform.")
                    # SFX: Action failed
            else:
                message_log.append("P1: Cannot terraform this tile.")
                # SFX: Action failed

        # --- Player 2 (AI) Turn ---
        self._opponent_turn(message_log)

        # --- End of Turn ---
        self.steps += 1
        self.message = " | ".join(message_log)

        terminated = self.steps >= self.MAX_TURNS
        if terminated:
            self.game_over = True
            p1_tiles, p2_tiles = self._count_tiles()
            if p1_tiles > p2_tiles:
                reward += 50
                self.score += 50
                self.message = f"GAME OVER: You win! ({p1_tiles} to {p2_tiles})"
            elif p2_tiles > p1_tiles:
                reward -= 50
                self.score -= 50
                self.message = f"GAME OVER: You lose. ({p1_tiles} to {p2_tiles})"
            else:
                self.message = f"GAME OVER: It's a draw! ({p1_tiles} to {p2_tiles})"
            # SFX: Game over fanfare

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _opponent_turn(self, message_log):
        # 1. Roll for Time Points
        p2_roll = self.np_random.integers(1, 7)
        self.p2_time_points += p2_roll
        message_log.append(f"P2 rolled {p2_roll}")

        # 2. AI Logic: Prioritize tech, then terraform
        # Try to research
        if self.p2_tech_level < self.TECH_LEVELS:
            cost = self.TECH_COSTS[self.p2_tech_level]
            if self.p2_time_points >= cost:
                self.p2_time_points -= cost
                self.p2_tech_level += 1
                message_log.append("P2 researched tech!")
                return
        
        # Try to terraform
        cost = self.TERRAFORM_COST
        if self.p2_tech_level >= 1:
            cost = max(1, cost - 1)
        
        if self.p2_time_points >= cost:
            barren_tiles = [pos for pos, tile in self.grid.items() if tile['type'] == self.TILE_BARREN]
            if barren_tiles:
                # Tech bonus: Level 2 lets AI terraform adjacent to its own tiles
                if self.p2_tech_level >= 2:
                    owned_tiles = {pos for pos, tile in self.grid.items() if tile['owner'] == self.PLAYER_2}
                    if owned_tiles:
                        candidate_tiles = []
                        for q, r in barren_tiles:
                            for dq, dr in [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]:
                                if (q+dq, r+dr) in owned_tiles:
                                    candidate_tiles.append((q,r))
                                    break
                        if candidate_tiles:
                            barren_tiles = candidate_tiles

                tile_to_terraform = barren_tiles[self.np_random.integers(len(barren_tiles))]
                pos = tile_to_terraform
                self.grid[pos]['type'] = self.TILE_FERTILE
                self.grid[pos]['owner'] = self.PLAYER_2
                self.p2_time_points -= cost
                message_log.append("P2 terraformed a tile.")
            else:
                message_log.append("P2: No barren tiles left.")

    def _move_cursor(self, movement):
        q, r = self.cursor_pos
        if movement == 1: r -= 1  # Up
        elif movement == 2: r += 1  # Down
        elif movement == 3: q -= 1  # Left
        elif movement == 4: q += 1  # Right
        
        # Clamp to grid bounds
        q = max(0, min(q, self.GRID_WIDTH - 1))
        r = max(0, min(r, self.GRID_HEIGHT - 1))
        
        # Adjust for offset grid layout
        if q % 2 != 0:
            r = max(0, min(r, self.GRID_HEIGHT - 2))
        
        self.cursor_pos = (q, r)
        
    def _generate_grid(self):
        self.grid = {}
        for q in range(self.GRID_WIDTH):
            r_limit = self.GRID_HEIGHT if q % 2 == 0 else self.GRID_HEIGHT - 1
            for r in range(r_limit):
                roll = self.np_random.random()
                tile_type = self.TILE_BARREN
                if roll > 0.9:
                    tile_type = self.TILE_MOUNTAIN
                elif roll > 0.8:
                    tile_type = self.TILE_WATER
                
                self.grid[(q, r)] = {'type': tile_type, 'owner': None}

    def _count_tiles(self):
        p1_tiles = 0
        p2_tiles = 0
        for tile in self.grid.values():
            if tile['type'] == self.TILE_FERTILE:
                if tile['owner'] == self.PLAYER_1:
                    p1_tiles += 1
                elif tile['owner'] == self.PLAYER_2:
                    p2_tiles += 1
        return p1_tiles, p2_tiles

    def _get_info(self):
        p1_tiles, p2_tiles = self._count_tiles()
        return {
            "score": self.score,
            "steps": self.steps,
            "p1_fertile_tiles": p1_tiles,
            "p2_fertile_tiles": p2_tiles,
            "p1_tech_level": self.p1_tech_level,
            "p2_tech_level": self.p2_tech_level,
            "p1_time_points": self.p1_time_points,
            "p2_time_points": self.p2_time_points,
            "cursor_pos": self.cursor_pos,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---
    
    def _render_game(self):
        # Render all hexes
        for (q, r), tile in self.grid.items():
            self._draw_hex(q, r, tile)
        
        # Render cursor
        self._draw_hex_cursor(self.cursor_pos)

    def _render_ui(self):
        # Draw UI background panels
        pygame.gfxdraw.box(self.screen, (0, 0, self.SCREEN_WIDTH, 40), self.COLOR_UI_BG)
        pygame.gfxdraw.box(self.screen, (0, self.SCREEN_HEIGHT - 30, self.SCREEN_WIDTH, 30), self.COLOR_UI_BG)

        # Top Bar: Turn and Score
        turn_text = f"Turn: {self.steps}/{self.MAX_TURNS}"
        self._draw_text(turn_text, (10, 10), self.font_medium)
        
        p1_tiles, p2_tiles = self._count_tiles()
        score_text = f"P1 Tiles: {p1_tiles} | P2 Tiles: {p2_tiles}"
        self._draw_text(score_text, (self.SCREEN_WIDTH - 250, 10), self.font_medium)
        
        # Player Info Panels
        self._render_player_info(self.PLAYER_1, (10, 50))
        self._render_player_info(self.PLAYER_2, (self.SCREEN_WIDTH - 210, 50))
        
        # Bottom Bar: Message Log
        self._draw_text(self.message, (10, self.SCREEN_HEIGHT - 22), self.font_small)

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            self._draw_text(self.message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_large, center=True)


    def _render_player_info(self, player_id, pos):
        x, y = pos
        color = self.PLAYER_COLORS[player_id]
        
        title = f"Player {player_id} (Agent)" if player_id == self.PLAYER_1 else f"Player {player_id} (AI)"
        self._draw_text(title, (x, y), self.font_medium, color)
        
        time_points = self.p1_time_points if player_id == self.PLAYER_1 else self.p2_time_points
        tech_level = self.p1_tech_level if player_id == self.PLAYER_1 else self.p2_tech_level
        
        self._draw_text(f"Time Points: {time_points}", (x + 10, y + 25), self.font_small)
        
        # Tech Level Bar
        self._draw_text("Tech Level:", (x + 10, y + 45), self.font_small)
        bar_x, bar_y, bar_w, bar_h = x + 10, y + 65, 150, 15
        pygame.draw.rect(self.screen, (50, 60, 80), (bar_x, bar_y, bar_w, bar_h))
        fill_w = (tech_level / self.TECH_LEVELS) * bar_w
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    # --- Hex Grid Drawing Logic ---
    
    def _hex_to_pixel(self, q, r):
        """Converts axial hex coordinates to pixel coordinates."""
        x_offset = 50
        y_offset = 120
        x = self.HEX_RADIUS * 3/2 * q + x_offset
        y = self.HEX_RADIUS * math.sqrt(3) * (r + q/2) + y_offset
        return int(x), int(y)

    def _draw_hex(self, q, r, tile):
        center_x, center_y = self._hex_to_pixel(q, r)
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((
                int(center_x + self.HEX_RADIUS * math.cos(angle_rad)),
                int(center_y + self.HEX_RADIUS * math.sin(angle_rad))
            ))

        # Draw main hex body
        main_color = self.TILE_TYPE_COLORS[tile['type']]
        pygame.gfxdraw.filled_polygon(self.screen, points, main_color)
        
        # Draw owner indicator if fertile
        if tile['type'] == self.TILE_FERTILE and tile['owner'] is not None:
            owner_color = self.PLAYER_COLORS[tile['owner']]
            inner_radius = self.HEX_RADIUS * 0.6
            inner_points = []
            for i in range(6):
                angle_deg = 60 * i + 30 # Offset to make a diamond-like shape
                angle_rad = math.pi / 180 * angle_deg
                inner_points.append((
                    int(center_x + inner_radius * math.cos(angle_rad)),
                    int(center_y + inner_radius * math.sin(angle_rad))
                ))
            pygame.gfxdraw.filled_polygon(self.screen, inner_points, owner_color)

        # Draw hex outline
        pygame.gfxdraw.aapolygon(self.screen, points, (0, 0, 0, 100))

    def _draw_hex_cursor(self, pos):
        q, r = pos
        center_x, center_y = self._hex_to_pixel(q, r)
        points = []
        # Use a slightly larger radius for the cursor so it stands out
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((
                int(center_x + (self.HEX_RADIUS + 1) * math.cos(angle_rad)),
                int(center_y + (self.HEX_RADIUS + 1) * math.sin(angle_rad))
            ))
        
        # Draw multiple lines for thickness
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # The main loop needs a visible display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the rendered frames
    pygame.display.set_caption("TerraHex Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop for human player
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not done:
            action = [movement, space, shift]
            if any(action): # Only step if an action was taken
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")
        
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control human play speed

    env.close()