
# Generated: 2025-08-27T14:55:13.250917
# Source Brief: brief_00832.md
# Brief Index: 832

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import re
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the crystal one tile at a time. "
        "Each move costs 1 point."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based isometric puzzle game. Guide the crystal through 10 rooms by activating "
        "mechanisms to open paths. You have a limited number of moves to solve all puzzles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 72)
        self.colors = {
            "bg": (20, 30, 40),
            "floor": (60, 70, 80),
            "wall_top": (100, 110, 120),
            "wall_front": (80, 90, 100),
            "crystal_outer": (100, 200, 255),
            "crystal_inner": (200, 240, 255),
            "exit": (0, 255, 150),
            "mechanism_off": (255, 120, 0),
            "mechanism_on": (255, 200, 50),
            "text": (230, 230, 230),
            "text_shadow": (10, 10, 10),
        }
        
        # Game constants
        self.total_rooms = 10
        self.total_moves = 200
        self.max_steps = 1000

        # Isometric projection parameters
        self.tile_width = 50
        self.tile_height = self.tile_width * 0.5
        self.wall_height = self.tile_height * 1.5

        # Room definitions
        self._define_rooms()
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def _define_rooms(self):
        # P: Player, E: Exit, W: Wall, ' ': Floor
        # M#: Mechanism, D#: Door linked to M#
        self.room_definitions = [
            # 1. Simple straight line
            ["WWWWW", "WP EW", "WWWWW"],
            # 2. Simple turn
            ["WWWWW", "W   W", "WP WWE", "WWWWW"],
            # 3. First mechanism
            ["WWWWWWW", "W  E  W", "W D1  W", "W M1  W", "W  P  W", "WWWWWWW"],
            # 4. Two mechanisms in a line
            ["WWWWWWWW", "W P D1 W", "W W M1 W", "W W    W", "W D2   W", "W M2 E W", "WWWWWWWW"],
            # 5. Choice of path
            ["WWWWWWWWW", "W E D1  W", "W W M1  W", "W WWWWWWW", "W P M2  W", "W   D2  W", "WWWWWWWWW"],
            # 6. Backtracking
            ["WWWWWWWWW", "WE D2 M2W", "W WWWWWWW", "W M1 P D1W", "W WWWWWWW"],
            # 7. Multiple doors for one switch
            ["WWWWWWWWWWW", "W P M1D1D1W", "W         W", "WWWWWD1WWWW", "W E       W", "WWWWWWWWWWW"],
            # 8. Complex pathing
            ["WWWWWWWWWWW", "W P M1 D2 W", "W W WWWWW W", "W D1  M3  W", "W WWW D3W W", "W M2  E   W", "WWWWWWWWWWW"],
            # 9. Key and lockbox
            ["WWWWWWWWWWW", "WP  M1    W", "W WWWWWD1WW", "W   M2    W", "W WWWD2WWWW", "W D3  E   W", "W M3      W", "WWWWWWWWWWW"],
            # 10. Final challenge
            ["WWWWWWWWWWWWW", "W P M1 D2   W", "W W WWWWW D3W", "W D1  M2    W", "W   WWWWWWW W", "WWWD3 M4  D4W", "W     M5    W", "W D5WWWWWWWWW", "W     E     W", "WWWWWWWWWWWWW"],
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.moves_remaining = self.total_moves
        self.current_room_index = 0
        
        self._load_room(self.current_room_index)
        
        return self._get_observation(), self._get_info()

    def _load_room(self, room_index):
        self.mechanisms = {}
        room_data = self.room_definitions[room_index]
        self.grid_height = len(room_data)
        self.grid_width = max(len(row) for row in room_data)
        
        self.grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        for r, row_str in enumerate(room_data):
            for c, char in enumerate(row_str):
                self.grid[r][c] = char
                if char == 'P':
                    self.crystal_pos = np.array([c, r])
                    self.grid[r][c] = ' '
                elif char == 'E':
                    self.exit_pos = np.array([c, r])
                elif char.startswith('M'):
                    num = int(re.search(r'\d+', char).group())
                    if num not in self.mechanisms: self.mechanisms[num] = {}
                    self.mechanisms[num]['pos'] = np.array([c, r])
                    self.mechanisms[num]['active'] = False
                elif char.startswith('D'):
                    num = int(re.search(r'\d+', char).group())
                    if num not in self.mechanisms: self.mechanisms[num] = {}
                    if 'targets' not in self.mechanisms[num]: self.mechanisms[num]['targets'] = []
                    self.mechanisms[num]['targets'].append(np.array([c, r]))
    
    def step(self, action):
        movement = action[0]
        reward = 0.0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        if movement != 0:
            if self.moves_remaining > 0:
                self.moves_remaining -= 1
                reward -= 0.1 # Cost of moving

                # 1=up(NW), 2=down(SE), 3=left(SW), 4=right(NE)
                move_map = {
                    1: np.array([0, -1]), # NW
                    2: np.array([0, 1]),  # SE
                    3: np.array([-1, 0]), # SW
                    4: np.array([1, 0])   # NE
                }
                delta = move_map[movement]
                target_pos = self.crystal_pos + delta

                if 0 <= target_pos[0] < self.grid_width and 0 <= target_pos[1] < self.grid_height:
                    target_tile = self.grid[target_pos[1]][target_pos[0]]
                    if target_tile not in ['W'] and not target_tile.startswith('D'):
                        self.crystal_pos = target_pos
                        # # Player moved sound
                
                # Check for mechanism activation
                for num, mech in self.mechanisms.items():
                    if 'pos' in mech and np.array_equal(self.crystal_pos, mech['pos']):
                        if not mech['active']:
                            mech['active'] = True
                            reward += 1.0
                            # # Mechanism activated sound
                            if 'targets' in mech:
                                for door_pos in mech['targets']:
                                    self.grid[door_pos[1]][door_pos[0]] = ' ' # Open door

                # Check for exit
                if np.array_equal(self.crystal_pos, self.exit_pos):
                    if self.current_room_index == self.total_rooms - 1:
                        self.game_over = True
                        self.victory = True
                        reward += 100.0
                        # # Victory sound
                    else:
                        reward += 10.0
                        self.current_room_index += 1
                        self._load_room(self.current_room_index)
                        # # Room complete sound
        
        self.steps += 1
        self.score += reward

        if self.moves_remaining <= 0 or self.steps >= self.max_steps:
            self.game_over = True
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _iso_to_cart(self, iso_pos):
        # Center the grid on the screen
        grid_pixel_width = (self.grid_width + self.grid_height) * self.tile_width / 2
        grid_pixel_height = (self.grid_width + self.grid_height) * self.tile_height / 2
        offset_x = (self.screen_width - grid_pixel_width) / 2 + self.grid_width * self.tile_width / 2
        offset_y = (self.screen_height - grid_pixel_height) / 2 + self.tile_height

        cart_x = (iso_pos[0] - iso_pos[1]) * (self.tile_width / 2) + offset_x
        cart_y = (iso_pos[0] + iso_pos[1]) * (self.tile_height / 2) + offset_y
        return int(cart_x), int(cart_y)

    def _draw_iso_tile(self, pos, top_color, front_color, height):
        center_x, top_y = self._iso_to_cart(pos)
        points = [
            (center_x, top_y),
            (center_x + self.tile_width / 2, top_y + self.tile_height / 2),
            (center_x, top_y + self.tile_height),
            (center_x - self.tile_width / 2, top_y + self.tile_height / 2)
        ]
        pygame.draw.polygon(self.screen, top_color, points)
        
        # Draw front faces for height
        if height > 0:
            p1 = points[2]
            p2 = points[3]
            p3 = (p2[0], p2[1] + height)
            p4 = (p1[0], p1[1] + height)
            pygame.draw.polygon(self.screen, front_color, [p1, p2, p3, p4])
            
            p1 = points[1]
            p2 = points[2]
            p3 = (p2[0], p2[1] + height)
            p4 = (p1[0], p1[1] + height)
            pygame.draw.polygon(self.screen, front_color, [p1, p2, p3, p4])

    def _draw_text(self, text, font, pos, color, shadow_color):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_observation(self):
        self.screen.fill(self.colors["bg"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw floor and static elements
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                pos = np.array([c, r])
                tile = self.grid[r][c]
                
                if tile == 'W':
                    self._draw_iso_tile(pos, self.colors["wall_top"], self.colors["wall_front"], self.wall_height)
                elif tile.startswith('D'):
                    self._draw_iso_tile(pos, self.colors["wall_top"], self.colors["wall_front"], self.wall_height)
                else: # Floor, mechanisms, exit
                    self._draw_iso_tile(pos, self.colors["floor"], self.colors["floor"], 0)
                    
                    if np.array_equal(pos, self.exit_pos):
                        cx, cy = self._iso_to_cart(pos)
                        cy += self.tile_height / 2
                        pygame.gfxdraw.filled_circle(self.screen, cx, int(cy), int(self.tile_width/3), self.colors["exit"])
                        pygame.gfxdraw.aacircle(self.screen, cx, int(cy), int(self.tile_width/3), self.colors["exit"])

                    if tile.startswith('M'):
                        num = int(re.search(r'\d+', tile).group())
                        mech = self.mechanisms[num]
                        cx, cy = self._iso_to_cart(pos)
                        cy += self.tile_height / 2
                        color = self.colors["mechanism_on"] if mech['active'] else self.colors["mechanism_off"]
                        pygame.gfxdraw.filled_circle(self.screen, cx, int(cy), int(self.tile_width/4), color)
                        pygame.gfxdraw.aacircle(self.screen, cx, int(cy), int(self.tile_width/4), color)

        # Draw crystal
        cx, cy = self._iso_to_cart(self.crystal_pos)
        cy += self.tile_height / 2 - self.wall_height # Make it appear on top of floor
        
        # Glow effect
        glow_radius = int(self.tile_width / 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.colors["crystal_outer"], 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (cx - glow_radius, cy - glow_radius))

        # Crystal body
        pygame.gfxdraw.filled_circle(self.screen, cx, int(cy), int(self.tile_width/3.5), self.colors["crystal_outer"])
        pygame.gfxdraw.aacircle(self.screen, cx, int(cy), int(self.tile_width/3.5), self.colors["crystal_outer"])
        pygame.gfxdraw.filled_circle(self.screen, cx, int(cy), int(self.tile_width/6), self.colors["crystal_inner"])
        pygame.gfxdraw.aacircle(self.screen, cx, int(cy), int(self.tile_width/6), self.colors["crystal_inner"])

    def _render_ui(self):
        # Room counter
        room_text = f"Room: {self.current_room_index + 1} / {self.total_rooms}"
        self._draw_text(room_text, self.font_ui, (10, 10), self.colors["text"], self.colors["text_shadow"])

        # Moves counter
        moves_text = f"Moves: {self.moves_remaining}"
        text_width = self.font_ui.size(moves_text)[0]
        self._draw_text(moves_text, self.font_ui, (self.screen_width - text_width - 10, 10), self.colors["text"], self.colors["text_shadow"])

        # Game Over / Victory message
        if self.game_over:
            if self.victory:
                msg = "VICTORY!"
            else:
                msg = "GAME OVER"
            
            text_width, text_height = self.font_msg.size(msg)
            pos = (self.screen_width/2 - text_width/2, self.screen_height/2 - text_height/2)
            self._draw_text(msg, self.font_msg, pos, self.colors["exit" if self.victory else "mechanism_off"], self.colors["text_shadow"])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "current_room": self.current_room_index + 1,
            "victory": self.victory,
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a Pygame window to display the environment
    pygame.display.set_caption("Crystal Quest")
    screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False

                # Only register a move if the game is not over
                if not terminated:
                    # 1=up(NW), 2=down(SE), 3=left(SW), 4=right(NE)
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the screen
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()