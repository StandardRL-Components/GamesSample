
# Generated: 2025-08-28T04:46:14.055504
# Source Brief: brief_02417.md
# Brief Index: 2417

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move. Press space to interact with objects."
    )

    game_description = (
        "Escape a procedurally generated cabin by solving puzzles before a lurking monster finds you."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game Constants
        self.TILE_SIZE = 40
        self.ROOM_W, self.ROOM_H = self.WIDTH // self.TILE_SIZE, self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 2000
        self.CABIN_GRID_SIZE = 5

        # Colors
        self.COLOR_BG = (10, 8, 12)
        self.COLOR_WALL = (40, 35, 45)
        self.COLOR_FLOOR = (25, 20, 30)
        self.COLOR_PLAYER = (255, 220, 180)
        self.COLOR_MONSTER = (180, 40, 60)
        self.COLOR_INTERACT = (80, 255, 80)
        self.COLOR_PUZZLE = (80, 120, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_DANGER = (255, 0, 0)
        self.COLOR_SUCCESS = (0, 255, 0)

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.monster_pos = None
        self.cabin_layout = None
        self.room_data = None
        self.puzzles = None
        self.puzzles_solved = 0
        self.exit_pos = None
        self.exit_open = False
        self.monster_patrol_cooldown = 0
        self.visited_tiles = None
        self.particles = None
        self.last_reward_text = None
        self.last_reward_timer = 0
        
        self.vignette_surface = self._create_vignette()

        self.reset()
        
        # This check is for development; comment out for production if needed
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.puzzles_solved = 0
        self.exit_open = False
        self.particles = []
        self.last_reward_text = None
        self.last_reward_timer = 0

        self._generate_cabin()
        self._place_entities()
        
        self.visited_tiles = {self.player_pos}

        return self._get_observation(), self._get_info()

    def _generate_cabin(self):
        self.cabin_layout = np.zeros((self.CABIN_GRID_SIZE, self.CABIN_GRID_SIZE), dtype=int)
        self.room_data = {}

        start_room = (self.CABIN_GRID_SIZE // 2, self.CABIN_GRID_SIZE // 2)
        path = [start_room]
        self.cabin_layout[start_room] = 1

        current_room = start_room
        for _ in range(8): # Generate a path of rooms
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_room[0] + dx, current_room[1] + dy
                if 0 <= nx < self.CABIN_GRID_SIZE and 0 <= ny < self.CABIN_GRID_SIZE and self.cabin_layout[nx, ny] == 0:
                    neighbors.append((nx, ny))
            if not neighbors:
                break
            next_room = self.np_random.choice([i for i in range(len(neighbors))])
            current_room = neighbors[next_room]
            self.cabin_layout[current_room] = 1
            path.append(current_room)
        
        self.start_room_pos = path[0]
        puzzle_rooms = self.np_random.choice([i for i in range(1, len(path)-1)], 3, replace=False)
        puzzle_room_coords = [path[i] for i in puzzle_rooms]
        self.exit_pos = path[-1]

        self._initialize_puzzles(puzzle_room_coords)
        
        for r in range(self.CABIN_GRID_SIZE):
            for c in range(self.CABIN_GRID_SIZE):
                if self.cabin_layout[r, c] == 1:
                    self._generate_room_layout((r, c))

    def _generate_room_layout(self, room_pos):
        layout = np.zeros((self.ROOM_W, self.ROOM_H), dtype=int) # 0: floor, 1: wall
        layout[0, :] = 1
        layout[-1, :] = 1
        layout[:, 0] = 1
        layout[:, -1] = 1
        
        # Add doors
        r, c = room_pos
        mid_x, mid_y = self.ROOM_W // 2, self.ROOM_H // 2
        if r > 0 and self.cabin_layout[r-1, c] == 1: layout[mid_x-1:mid_x+1, 0] = 0 # Up
        if r < self.CABIN_GRID_SIZE - 1 and self.cabin_layout[r+1, c] == 1: layout[mid_x-1:mid_x+1, -1] = 0 # Down
        if c > 0 and self.cabin_layout[r, c-1] == 1: layout[0, mid_y-1:mid_y+1] = 0 # Left
        if c < self.CABIN_GRID_SIZE - 1 and self.cabin_layout[r, c+1] == 1: layout[-1, mid_y-1:mid_y+1] = 0 # Right
        
        self.room_data[room_pos] = {'layout': layout}

    def _initialize_puzzles(self, puzzle_rooms):
        self.puzzles = []
        # Puzzle 1: Levers
        solution = tuple(self.np_random.integers(0, 2, size=3))
        self.puzzles.append({
            'type': 'levers', 'room': puzzle_rooms[0], 'solved': False,
            'state': [0, 0, 0], 'solution': solution,
            'pos': [(self.ROOM_W//2 - 2, 2), (self.ROOM_W//2, 2), (self.ROOM_W//2 + 2, 2)]
        })
        # Puzzle 2: Pressure Plates
        plate_pos = self.np_random.choice(self.ROOM_W * self.ROOM_H, 5, replace=False)
        plate_pos = [(p % self.ROOM_W, p // self.ROOM_W) for p in plate_pos]
        self.puzzles.append({
            'type': 'plates', 'room': puzzle_rooms[1], 'solved': False,
            'state': [False] * 5,
            'pos': plate_pos
        })
        # Puzzle 3: Sequence
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        self.np_random.shuffle(colors)
        solution = tuple(colors)
        statue_pos = [(4, 4), (self.ROOM_W//2, self.ROOM_H-4), (self.ROOM_W-4, 4)]
        self.puzzles.append({
            'type': 'sequence', 'room': puzzle_rooms[2], 'solved': False,
            'state': [], 'solution': solution,
            'pos': statue_pos, 'colors': [(255,0,0), (0,255,0), (0,0,255)]
        })

    def _place_entities(self):
        self.player_pos = (*self.start_room_pos, self.ROOM_W // 2, self.ROOM_H // 2)
        
        possible_monster_starts = []
        for r in range(self.CABIN_GRID_SIZE):
            for c in range(self.CABIN_GRID_SIZE):
                if self.cabin_layout[r,c] == 0:
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < self.CABIN_GRID_SIZE and 0 <= nc < self.CABIN_GRID_SIZE and self.cabin_layout[nr,nc] == 1:
                            possible_monster_starts.append((nr, nc))
                            break
        
        monster_start_room = random.choice(possible_monster_starts) if possible_monster_starts else self.exit_pos
        self.monster_pos = (*monster_start_room, self.ROOM_W // 2, self.ROOM_H // 2)
        self.monster_patrol_cooldown = 10


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, _ = action
        
        # Handle player actions
        if movement > 0:
            reward += self._handle_player_movement(movement)
        if space_held:
            reward += self._handle_interaction()

        # Update monster
        self._update_monster()

        # Check for game end conditions
        terminated = self._check_termination()
        if terminated:
            if self.player_pos[:2] == self.exit_pos and self.exit_open:
                reward += 100
                self._add_reward_text("+100 ESCAPED!", self.COLOR_SUCCESS)
            else: # Caught by monster or timeout
                reward -= 100
                self._add_reward_text("-100 CAUGHT!", self.COLOR_DANGER)
            self.game_over = True
        
        # Update score
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement):
        r, c, x, y = self.player_pos
        
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        nx, ny = x + dx, y + dy
        
        current_room_layout = self.room_data[(r, c)]['layout']

        # Room transition
        if not (0 <= nx < self.ROOM_W and 0 <= ny < self.ROOM_H):
            nr, nc = r, c
            if ny < 0 and r > 0 and self.cabin_layout[r-1, c] == 1: nr, ny = r-1, self.ROOM_H - 1
            elif ny >= self.ROOM_H and r < self.CABIN_GRID_SIZE - 1 and self.cabin_layout[r+1, c] == 1: nr, ny = r+1, 0
            elif nx < 0 and c > 0 and self.cabin_layout[r, c-1] == 1: nc, nx = c-1, self.ROOM_W - 1
            elif nx >= self.ROOM_W and c < self.CABIN_GRID_SIZE - 1 and self.cabin_layout[r, c+1] == 1: nc, nx = c+1, 0
            
            if (nr, nc) != (r, c):
                self.player_pos = (nr, nc, nx, ny)
        # Movement within room
        elif current_room_layout[nx, ny] == 0:
            self.player_pos = (r, c, nx, ny)

        # Exploration reward
        if self.player_pos not in self.visited_tiles:
            self.visited_tiles.add(self.player_pos)
            return 0.1
        else:
            return -0.01

    def _handle_interaction(self):
        pr, pc, px, py = self.player_pos
        reward = 0
        
        # Check for puzzle interactions
        for puzzle in self.puzzles:
            if puzzle['room'] == (pr, pc) and not puzzle['solved']:
                for i, pos in enumerate(puzzle['pos']):
                    if abs(px - pos[0]) <= 1 and abs(py - pos[1]) <= 1:
                        # Puzzle interaction logic
                        if puzzle['type'] == 'levers':
                            puzzle['state'][i] = 1 - puzzle['state'][i]
                            if tuple(puzzle['state']) == puzzle['solution']:
                                puzzle['solved'] = True
                                reward += 5
                                self.puzzles_solved += 1
                                self._create_particles(pos, self.COLOR_SUCCESS)
                                self._add_reward_text("+5 Puzzle Solved!", self.COLOR_SUCCESS)
                        elif puzzle['type'] == 'sequence':
                            puzzle['state'].append(puzzle['colors'][i])
                            if tuple(puzzle['state']) == puzzle['solution'][:len(puzzle['state'])]:
                                self._create_particles(pos, puzzle['colors'][i])
                                if len(puzzle['state']) == len(puzzle['solution']):
                                    puzzle['solved'] = True
                                    reward += 5
                                    self.puzzles_solved += 1
                                    self._add_reward_text("+5 Puzzle Solved!", self.COLOR_SUCCESS)
                            else:
                                puzzle['state'] = [] # Reset on wrong sequence
                                self._create_particles(pos, self.COLOR_DANGER, 20)
                        return reward # Prevent multiple interactions in one step
        
        # Check for exit interaction
        if self.exit_open and (pr, pc) == self.exit_pos:
            exit_tile = (self.ROOM_W // 2, self.ROOM_H // 2)
            if abs(px - exit_tile[0]) <= 2 and abs(py - exit_tile[1]) <= 2:
                # Win condition is handled in _check_termination
                pass
        
        return reward

    def _update_monster(self):
        pr, pc, px, py = self.player_pos
        mr, mc, mx, my = self.monster_pos
        
        # If in same room, hunt player
        if (pr, pc) == (mr, mc):
            self.monster_patrol_cooldown = 10 # Reset patrol timer when hunting
            dx, dy = np.sign(px - mx), np.sign(py - my)
            
            # Simple pathfinding (move on one axis)
            if dx != 0 and self.room_data[(mr,mc)]['layout'][mx+dx, my] == 0:
                self.monster_pos = (mr, mc, mx + dx, my)
            elif dy != 0 and self.room_data[(mr,mc)]['layout'][mx, my+dy] == 0:
                self.monster_pos = (mr, mc, mx, my + dy)
        # If in different rooms, patrol
        else:
            self.monster_patrol_cooldown -= 1
            patrol_speed = max(1, 10 - self.steps // 200)
            if self.monster_patrol_cooldown <= 0:
                self.monster_patrol_cooldown = patrol_speed
                neighbors = []
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = mr + dr, mc + dc
                    if 0 <= nr < self.CABIN_GRID_SIZE and 0 <= nc < self.CABIN_GRID_SIZE and self.cabin_layout[nr, nc] == 1:
                        neighbors.append((nr, nc))
                
                if neighbors:
                    next_room = self.np_random.choice([i for i in range(len(neighbors))])
                    self.monster_pos = (*neighbors[next_room], self.ROOM_W // 2, self.ROOM_H // 2)

    def _check_termination(self):
        if self.puzzles_solved == 3 and not self.exit_open:
            self.exit_open = True
            self._add_reward_text("The Exit is Open!", self.COLOR_TEXT)
            self._create_particles((self.ROOM_W//2, self.ROOM_H//2), self.COLOR_SUCCESS, 50, room=self.exit_pos)

        pr, pc, _, _ = self.player_pos
        mr, mc, _, _ = self.monster_pos

        # Caught by monster
        if self.player_pos == self.monster_pos:
            return True
        
        # Reached exit
        if self.exit_open and (pr, pc) == self.exit_pos:
            return True
            
        # Max steps
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pr, pc, px, py = self.player_pos
        current_room_pos = (pr, pc)
        
        # Render floor and walls for current room
        layout = self.room_data[current_room_pos]['layout']
        for x in range(self.ROOM_W):
            for y in range(self.ROOM_H):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if layout[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Render puzzles
        for puzzle in self.puzzles:
            if puzzle['room'] == current_room_pos:
                if puzzle['type'] == 'levers':
                    for i, pos in enumerate(puzzle['pos']):
                        lever_rect = pygame.Rect(pos[0]*self.TILE_SIZE, pos[1]*self.TILE_SIZE, 8, self.TILE_SIZE)
                        pygame.draw.rect(self.screen, (100,100,100), lever_rect.inflate(8,8))
                        pygame.draw.rect(self.screen, self.COLOR_PUZZLE if not puzzle['solved'] else self.COLOR_SUCCESS, lever_rect)
                        handle_y = pos[1]*self.TILE_SIZE if puzzle['state'][i] == 0 else pos[1]*self.TILE_SIZE + self.TILE_SIZE - 8
                        pygame.draw.rect(self.screen, (200,200,200), (lever_rect.centerx-8, handle_y, 16, 8))
                elif puzzle['type'] == 'plates':
                    for i, pos in enumerate(puzzle['pos']):
                        is_active = (px, py) == pos
                        if is_active and not puzzle['state'][i]:
                            puzzle['state'][i] = True
                        color = self.COLOR_SUCCESS if puzzle['state'][i] else self.COLOR_DANGER
                        pygame.gfxdraw.box(self.screen, (pos[0]*self.TILE_SIZE, pos[1]*self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), (*color, 100))
                    if all(puzzle['state']) and not puzzle['solved']:
                        puzzle['solved'] = True
                        self.puzzles_solved += 1
                        self.score += 5
                        self._add_reward_text("+5 Puzzle Solved!", self.COLOR_SUCCESS)

                elif puzzle['type'] == 'sequence':
                    # Hint plaque
                    hint_rect = pygame.Rect(self.WIDTH - 100, 20, 80, 30)
                    pygame.draw.rect(self.screen, (50,50,50), hint_rect)
                    for i, color in enumerate(puzzle['solution']):
                        pygame.draw.rect(self.screen, color, (hint_rect.x + 5 + i*25, hint_rect.y + 5, 20, 20))
                    
                    for i, pos in enumerate(puzzle['pos']):
                        color = puzzle['colors'][i]
                        is_lit = color in puzzle['state']
                        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]*self.TILE_SIZE + self.TILE_SIZE/2), int(pos[1]*self.TILE_SIZE + self.TILE_SIZE/2), 15, color)
                        if is_lit:
                            pygame.gfxdraw.aacircle(self.screen, int(pos[0]*self.TILE_SIZE + self.TILE_SIZE/2), int(pos[1]*self.TILE_SIZE + self.TILE_SIZE/2), 18, (255,255,0))
                            
        # Render exit
        if current_room_pos == self.exit_pos:
            color = self.COLOR_SUCCESS if self.exit_open else self.COLOR_DANGER
            exit_rect = pygame.Rect((self.ROOM_W//2 - 1)*self.TILE_SIZE, (self.ROOM_H//2 - 1)*self.TILE_SIZE, self.TILE_SIZE*2, self.TILE_SIZE*2)
            pygame.draw.rect(self.screen, color, exit_rect, 3)
            text = self.font_small.render("EXIT", True, color)
            self.screen.blit(text, text.get_rect(center=exit_rect.center))
        
        # Render Player
        player_rect = pygame.Rect(px * self.TILE_SIZE, py * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-8, -8))
        pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(-8,-8), (*self.COLOR_PLAYER, 180))

        # Render Monster
        mr, mc, mx, my = self.monster_pos
        if (pr, pc) == (mr, mc):
            monster_rect = pygame.Rect(mx * self.TILE_SIZE, my * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_MONSTER, monster_rect.inflate(-4, -4))
            # Eyes
            eye_y = monster_rect.centery - 4
            pygame.draw.circle(self.screen, (255,255,255), (monster_rect.centerx - 5, eye_y), 3)
            pygame.draw.circle(self.screen, (255,255,255), (monster_rect.centerx + 5, eye_y), 3)

    def _render_effects(self):
        # Update and draw particles
        for p in self.particles[:]:
            p[0] += p[2] # pos_x += vel_x
            p[1] += p[3] # pos_y += vel_y
            p[5] -= 1 # lifetime
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p[5] / p[6]))))
                color = (*p[4], alpha)
                size = int(p[5] / p[6] * 5)
                if size > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

        # Monster proximity vignette
        dist = float('inf')
        pr, pc, _, _ = self.player_pos
        mr, mc, _, _ = self.monster_pos
        if (pr, pc) == (mr, mc):
             dist = 0
        else:
             # A* or simple path distance would be better, but Manhattan is ok for an effect
             dist = abs(pr - mr) + abs(pc - mc)
        
        vignette_alpha = int(max(0, min(200, 200 - dist * 40)))
        if vignette_alpha > 0:
            self.vignette_surface.set_alpha(vignette_alpha)
            self.screen.blit(self.vignette_surface, (0,0))
            
    def _create_vignette(self):
        vignette = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(vignette, (0, 0, 0, 255), (0, 0, self.WIDTH, self.HEIGHT))
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        radius = min(center_x, center_y) * 1.5
        for r in range(int(radius), 0, -1):
            alpha = 255 * (1 - r / radius)**2
            pygame.gfxdraw.filled_circle(vignette, center_x, center_y, r, (0, 0, 0, int(255 - alpha)))
        return vignette


    def _render_ui(self):
        # Puzzles solved
        text = self.font_small.render(f"Puzzles Solved: {self.puzzles_solved} / 3", True, self.COLOR_TEXT)
        self.screen.blit(text, (10, 10))
        
        # Steps
        text = self.font_small.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(text, (self.WIDTH - text.get_width() - 10, 10))

        # Reward popup
        if self.last_reward_timer > 0:
            alpha = max(0, min(255, int(255 * (self.last_reward_timer / 60))))
            color = (*self.last_reward_color, alpha)
            text_surf = self.font_large.render(self.last_reward_text, True, color)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH/2, 60)))
            self.last_reward_timer -= 1
        
        # Game Over text
        if self.game_over:
            msg = "YOU ESCAPED!" if self.exit_open and self.player_pos[:2] == self.exit_pos else "YOU WERE CAUGHT"
            color = self.COLOR_SUCCESS if msg == "YOU ESCAPED!" else self.COLOR_DANGER
            text = self.font_large.render(msg, True, color)
            self.screen.blit(text, text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "puzzles_solved": self.puzzles_solved,
            "player_pos": self.player_pos,
            "monster_pos": self.monster_pos,
        }

    def _create_particles(self, pos, color, count=30, room=None):
        if room is None:
            room = self.player_pos[:2]
        
        if room != self.player_pos[:2]:
            return
            
        center_x = pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifetime = self.np_random.integers(30, 60)
            self.particles.append([center_x, center_y, vel_x, vel_y, color, lifetime, lifetime])

    def _add_reward_text(self, text, color):
        self.last_reward_text = text
        self.last_reward_color = color
        self.last_reward_timer = 60

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    # For human play, we need a real screen
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cabin Escape")
    
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # no-op, no-space, no-shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if done:
            # If the game is over, wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                done = False
                total_reward = 0
        else:
            # Only step if an action is taken, because auto_advance is False
            if action != [0, 0, 0]:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Done: {done}")

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    env.close()