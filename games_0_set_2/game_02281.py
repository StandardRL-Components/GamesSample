
# Generated: 2025-08-28T04:19:11.636825
# Source Brief: brief_02281.md
# Brief Index: 2281

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to interact with objects (levers)."
    )

    game_description = (
        "Escape a procedurally generated haunted house by solving a puzzle to unlock the exit, "
        "all while evading patrolling ghosts. You have two lives."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_FURNITURE = (25, 25, 35)
        self.COLOR_PLAYER = (70, 170, 255)
        self.COLOR_GHOST = (255, 255, 255)
        self.COLOR_LEVER_OFF = (200, 50, 50)
        self.COLOR_LEVER_ON = (50, 200, 50)
        self.COLOR_LOCKED_DOOR = (180, 100, 20)
        self.COLOR_UNLOCKED_DOOR = (100, 180, 20)
        self.COLOR_EXIT_DOOR = (50, 255, 150)
        self.COLOR_UI_TEXT = (200, 200, 220)

        # Game constants
        self.GRID_SIZE = (5, 4)
        self.ROOM_WIDTH = self.width
        self.ROOM_HEIGHT = self.height
        self.WALL_THICKNESS = 20
        self.DOOR_WIDTH = 60
        self.DOOR_HEIGHT = 10
        self.MAX_STEPS = 1000
        self.MAX_ENCOUNTERS = 2
        
        # All state variables are initialized in reset()
        self.rooms = []
        self.player_pos = None
        self.player_room_coords = None
        self.player_size = 12
        self.player_invincible_timer = 0
        self.ghosts = {}
        self.start_room_coords = None
        self.exit_room_coords = None
        self.puzzle_room_coords = None
        self.locked_door_room_coords = None
        self.locked_door_side = None
        
        self.steps = 0
        self.score = 0
        self.ghost_encounters = 0
        self.ghost_speed = 1.0
        self.lever_pulled = False
        self.game_over_message = ""

        # Initialize state
        self.reset()
        
        self.validate_implementation()

    def _generate_house(self):
        rows, cols = self.GRID_SIZE
        self.rooms = [[{'neighbors': [], 'ghosts': [], 'furniture': []} for _ in range(cols)] for _ in range(rows)]
        
        # Use randomized DFS to create a maze-like structure
        stack = [(0, 0)]
        visited = set([(0, 0)])
        
        while stack:
            r, c = stack[-1]
            unvisited_neighbors = []
            for dr, dc, side, opposite_side in [(0, 1, 'right', 'left'), (0, -1, 'left', 'right'), (1, 0, 'down', 'up'), (-1, 0, 'up', 'down')]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    unvisited_neighbors.append((nr, nc, side, opposite_side))
            
            if unvisited_neighbors:
                nr, nc, side, opposite_side = random.choice(unvisited_neighbors)
                self.rooms[r][c]['neighbors'].append(side)
                self.rooms[nr][nc]['neighbors'].append(opposite_side)
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

    def _place_entities(self):
        rows, cols = self.GRID_SIZE
        
        # Place start and exit rooms far apart
        self.start_room_coords = (self.np_random.integers(0, rows), self.np_random.integers(0, cols // 2))
        self.exit_room_coords = (self.np_random.integers(0, rows), self.np_random.integers(cols // 2, cols))
        
        # Find a path for the puzzle
        path = self._find_room_path(self.start_room_coords, self.exit_room_coords)
        if not path or len(path) < 3:
             # If path is too short or non-existent, regenerate
            self.reset()
            return

        # Place locked door and lever on the path
        self.locked_door_room_coords = path[len(path) // 2]
        
        # Determine which door to lock on the path
        prev_room = path[path.index(self.locked_door_room_coords) - 1]
        dr = self.locked_door_room_coords[0] - prev_room[0]
        dc = self.locked_door_room_coords[1] - prev_room[1]
        
        if dr == 1: self.locked_door_side = 'up'
        elif dr == -1: self.locked_door_side = 'down'
        elif dc == 1: self.locked_door_side = 'left'
        else: self.locked_door_side = 'right'

        # Place lever in a room off the main path if possible, otherwise on it but not start/end
        possible_lever_rooms = [ (r,c) for r in range(rows) for c in range(cols) if (r,c) not in [self.start_room_coords, self.exit_room_coords, self.locked_door_room_coords]]
        if not possible_lever_rooms:
            possible_lever_rooms = [p for p in path if p not in [self.start_room_coords, self.exit_room_coords, self.locked_door_room_coords]]
            if not possible_lever_rooms: # Fallback if path is very short
                 possible_lever_rooms = [self.start_room_coords]

        self.puzzle_room_coords = random.choice(possible_lever_rooms)

        # Place ghosts and furniture
        for r in range(rows):
            for c in range(cols):
                # Add ghosts to non-critical rooms
                if (r, c) not in [self.start_room_coords, self.exit_room_coords, self.puzzle_room_coords] and self.np_random.random() < 0.3:
                    num_ghosts = self.np_random.integers(1, 3)
                    for _ in range(num_ghosts):
                        ghost_y = self.np_random.integers(self.WALL_THICKNESS + 20, self.height - self.WALL_THICKNESS - 20)
                        self.rooms[r][c]['ghosts'].append({
                            'pos': pygame.Vector2(self.width / 2, ghost_y),
                            'dir': self.np_random.choice([-1, 1]),
                            'size': 15
                        })
                # Add furniture
                if self.np_random.random() < 0.5:
                     self.rooms[r][c]['furniture'].append(pygame.Rect(
                         self.np_random.integers(100, self.width-200),
                         self.np_random.integers(100, self.height-200),
                         self.np_random.integers(50, 150),
                         self.np_random.integers(50, 150)
                     ))

    def _find_room_path(self, start, end):
        q = deque([[start]])
        visited = {start}
        while q:
            path = q.popleft()
            r, c = path[-1]
            if (r, c) == end:
                return path
            
            room = self.rooms[r][c]
            for neighbor_dir in room['neighbors']:
                dr, dc = 0, 0
                if neighbor_dir == 'up': dr = -1
                elif neighbor_dir == 'down': dr = 1
                elif neighbor_dir == 'left': dc = -1
                elif neighbor_dir == 'right': dc = 1
                
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    q.append(new_path)
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_house()
        self._place_entities()
        
        self.player_room_coords = self.start_room_coords
        self.player_pos = pygame.Vector2(self.width / 2, self.height / 2)
        self.player_invincible_timer = 0

        self.steps = 0
        self.score = 0
        self.ghost_encounters = 0
        self.ghost_speed = 1.0
        self.lever_pulled = False
        self.game_over = False
        self.game_over_message = ""
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.01  # Small penalty for taking time

        # --- Update game logic ---
        self.steps += 1
        if self.player_invincible_timer > 0:
            self.player_invincible_timer -= 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.ghost_speed += 0.05

        # 1. Handle player movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            self._move_player(move_vec * 3)

        # 2. Handle interaction
        if space_held == 1:
            if self.player_room_coords == self.puzzle_room_coords:
                lever_rect = pygame.Rect(self.width / 2 - 10, self.height / 2 - 25, 20, 50)
                if lever_rect.colliderect(self._get_player_rect()):
                    if not self.lever_pulled:
                        self.lever_pulled = True
                        reward += 5 # Reward for solving puzzle
                        # SFX: Lever pull
        
        # 3. Update ghosts in current room
        r, c = self.player_room_coords
        for ghost in self.rooms[r][c]['ghosts']:
            ghost['pos'].x += ghost['dir'] * self.ghost_speed
            if ghost['pos'].x < self.WALL_THICKNESS + ghost['size'] or ghost['pos'].x > self.width - self.WALL_THICKNESS - ghost['size']:
                ghost['dir'] *= -1
            
            ghost_rect = pygame.Rect(ghost['pos'].x - ghost['size'], ghost['pos'].y - ghost['size'], ghost['size']*2, ghost['size']*2)
            if ghost_rect.colliderect(self._get_player_rect()) and self.player_invincible_timer == 0:
                self.ghost_encounters += 1
                reward -= 10 # Penalty for encounter
                self.player_invincible_timer = 60 # 2 seconds of invincibility at 30fps
                # SFX: Ghost hit
        
        # 4. Check for termination
        terminated = False
        if self.ghost_encounters >= self.MAX_ENCOUNTERS:
            terminated = True
            reward -= 50
            self.game_over = True
            self.game_over_message = "CAUGHT!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME'S UP!"
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _move_player(self, move_vec):
        new_pos = self.player_pos + move_vec
        player_rect = pygame.Rect(new_pos.x - self.player_size / 2, new_pos.y - self.player_size / 2, self.player_size, self.player_size)
        
        # Room transition
        r, c = self.player_room_coords
        room = self.rooms[r][c]
        
        # Horizontal doors
        if 'up' in room['neighbors'] and player_rect.top < self.WALL_THICKNESS:
            if self._is_at_door(player_rect.centerx, 'horizontal'):
                if self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'up' and not self.lever_pulled: return
                self.player_room_coords = (r - 1, c)
                self.player_pos.y = self.height - self.WALL_THICKNESS - self.player_size
                return
        if 'down' in room['neighbors'] and player_rect.bottom > self.height - self.WALL_THICKNESS:
            if self._is_at_door(player_rect.centerx, 'horizontal'):
                if self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'down' and not self.lever_pulled: return
                self.player_room_coords = (r + 1, c)
                self.player_pos.y = self.WALL_THICKNESS + self.player_size
                return
        
        # Vertical doors
        if 'left' in room['neighbors'] and player_rect.left < self.WALL_THICKNESS:
            if self._is_at_door(player_rect.centery, 'vertical'):
                if self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'left' and not self.lever_pulled: return
                self.player_room_coords = (r, c - 1)
                self.player_pos.x = self.width - self.WALL_THICKNESS - self.player_size
                return
        if 'right' in room['neighbors'] and player_rect.right > self.width - self.WALL_THICKNESS:
            if self._is_at_door(player_rect.centery, 'vertical'):
                if self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'right' and not self.lever_pulled: return
                self.player_room_coords = (r, c + 1)
                self.player_pos.x = self.WALL_THICKNESS + self.player_size
                return

        # Check for exit
        if self.player_room_coords == self.exit_room_coords and 'right' not in room['neighbors'] and player_rect.right > self.width - self.WALL_THICKNESS:
             if self._is_at_door(player_rect.centery, 'vertical'):
                 self.score += 100
                 self.game_over = True
                 self.game_over_message = "ESCAPED!"
                 return

        # Wall collisions
        if new_pos.x < self.WALL_THICKNESS + self.player_size/2 or new_pos.x > self.width - self.WALL_THICKNESS - self.player_size/2:
            move_vec.x = 0
        if new_pos.y < self.WALL_THICKNESS + self.player_size/2 or new_pos.y > self.height - self.WALL_THICKNESS - self.player_size/2:
            move_vec.y = 0

        # Furniture collision
        for furn_rect in room['furniture']:
            # Predict collision and stop movement component
            temp_rect_x = pygame.Rect(self.player_pos.x + move_vec.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)
            if temp_rect_x.colliderect(furn_rect):
                move_vec.x = 0
            temp_rect_y = pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y + move_vec.y - self.player_size/2, self.player_size, self.player_size)
            if temp_rect_y.colliderect(furn_rect):
                move_vec.y = 0
        
        self.player_pos += move_vec

    def _is_at_door(self, pos, orientation):
        if orientation == 'horizontal':
            return self.width/2 - self.DOOR_WIDTH/2 < pos < self.width/2 + self.DOOR_WIDTH/2
        else: # vertical
            return self.height/2 - self.DOOR_WIDTH/2 < pos < self.height/2 + self.DOOR_WIDTH/2

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_darkness_and_light()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        r, c = self.player_room_coords
        room = self.rooms[r][c]
        
        # Walls and Doors
        self._draw_walls_and_doors(room)

        # Furniture
        for furn_rect in room['furniture']:
            pygame.draw.rect(self.screen, self.COLOR_FURNITURE, furn_rect)
        
        # Puzzle Lever
        if (r,c) == self.puzzle_room_coords:
            lever_rect = pygame.Rect(self.width / 2 - 5, self.height / 2 - 20, 10, 40)
            color = self.COLOR_LEVER_ON if self.lever_pulled else self.COLOR_LEVER_OFF
            pygame.draw.rect(self.screen, color, lever_rect)
            # Pulsating highlight for interactive object
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 10
            highlight_rect = lever_rect.inflate(pulse, pulse)
            pygame.draw.rect(self.screen, color, highlight_rect, 1)

        # Ghosts
        ghost_bob = math.sin(self.steps * 0.2) * 4
        for ghost in room['ghosts']:
            alpha = 150 + math.sin(self.steps * 0.3 + ghost['pos'].y) * 50
            s = pygame.Surface((ghost['size']*2, ghost['size']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, ghost['size'], ghost['size'], ghost['size'], (*self.COLOR_GHOST, int(alpha)))
            pygame.gfxdraw.aacircle(s, ghost['size'], ghost['size'], ghost['size'], (*self.COLOR_GHOST, int(alpha)))
            self.screen.blit(s, (int(ghost['pos'].x - ghost['size']), int(ghost['pos'].y - ghost['size'] + ghost_bob)))
            
        # Player
        if self.player_invincible_timer > 0 and self.steps % 4 < 2:
            pass # Flicker when invincible
        else:
            player_rect = self._get_player_rect()
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), int(self.player_size*0.7), (*self.COLOR_PLAYER, 100))
    
    def _draw_walls_and_doors(self, room):
        wt = self.WALL_THICKNESS
        dw = self.DOOR_WIDTH
        
        # Top wall
        if 'up' in room['neighbors']:
            is_locked = self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'up' and not self.lever_pulled
            color = self.COLOR_LOCKED_DOOR if is_locked else self.COLOR_WALL
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.width/2 - dw/2, wt))
            pygame.draw.rect(self.screen, color, (self.width/2 - dw/2, 0, dw, self.DOOR_HEIGHT))
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width/2 + dw/2, 0, self.width/2 - dw/2, wt))
        else:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.width, wt))
            
        # Bottom wall
        if 'down' in room['neighbors']:
            is_locked = self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'down' and not self.lever_pulled
            color = self.COLOR_LOCKED_DOOR if is_locked else self.COLOR_WALL
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.height-wt, self.width/2 - dw/2, wt))
            pygame.draw.rect(self.screen, color, (self.width/2 - dw/2, self.height - self.DOOR_HEIGHT, dw, self.DOOR_HEIGHT))
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width/2 + dw/2, self.height-wt, self.width/2 - dw/2, wt))
        else:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.height-wt, self.width, wt))

        # Left wall
        if 'left' in room['neighbors']:
            is_locked = self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'left' and not self.lever_pulled
            color = self.COLOR_LOCKED_DOOR if is_locked else self.COLOR_WALL
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, wt, self.height/2 - dw/2))
            pygame.draw.rect(self.screen, color, (0, self.height/2 - dw/2, self.DOOR_HEIGHT, dw))
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.height/2 + dw/2, wt, self.height/2 - dw/2))
        else:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, wt, self.height))
            
        # Right wall
        if self.player_room_coords == self.exit_room_coords and 'right' not in room['neighbors']:
            # This is the exit door
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width-wt, 0, wt, self.height/2 - dw/2))
            pygame.draw.rect(self.screen, self.COLOR_EXIT_DOOR, (self.width-self.DOOR_HEIGHT, self.height/2 - dw/2, self.DOOR_HEIGHT, dw))
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width-wt, self.height/2 + dw/2, wt, self.height/2 - dw/2))
        elif 'right' in room['neighbors']:
            is_locked = self.player_room_coords == self.locked_door_room_coords and self.locked_door_side == 'right' and not self.lever_pulled
            color = self.COLOR_LOCKED_DOOR if is_locked else self.COLOR_WALL
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width-wt, 0, wt, self.height/2 - dw/2))
            pygame.draw.rect(self.screen, color, (self.width-self.DOOR_HEIGHT, self.height/2 - dw/2, self.DOOR_HEIGHT, dw))
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width-wt, self.height/2 + dw/2, wt, self.height/2 - dw/2))
        else:
            pygame.draw.rect(self.screen, self.COLOR_WALL, (self.width-wt, 0, wt, self.height))

    def _render_darkness_and_light(self):
        dark_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dark_surf.fill((0,0,0, 220))
        
        flicker = self.np_random.integers(-10, 11)
        light_radius = 120 + flicker + math.sin(self.steps * 0.05) * 10
        
        pygame.draw.circle(dark_surf, (0,0,0,0), (int(self.player_pos.x), int(self.player_pos.y)), int(light_radius))
        self.screen.blit(dark_surf, (0,0))
        
    def _render_ui(self):
        path = self._find_room_path(self.player_room_coords, self.exit_room_coords)
        dist = len(path) - 1 if path else 99

        encounters_text = self.font_ui.render(f"Ghost Encounters: {self.ghost_encounters}/{self.MAX_ENCOUNTERS}", True, self.COLOR_UI_TEXT)
        dist_text = self.font_ui.render(f"Distance to Exit: {dist} rooms", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(encounters_text, (10, 10))
        self.screen.blit(dist_text, (10, 30))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_game_over.render(self.game_over_message, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ghost_encounters": self.ghost_encounters,
            "distance_to_exit": len(self._find_room_path(self.player_room_coords, self.exit_room_coords) or []) - 1,
            "lever_pulled": self.lever_pulled,
        }

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