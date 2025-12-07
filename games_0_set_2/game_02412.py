
# Generated: 2025-08-27T20:19:18.366247
# Source Brief: brief_02412.md
# Brief Index: 2412

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Player:
    def __init__(self, x, y):
        self.grid_x, self.grid_y = x, y
        self.world_x, self.world_y = float(x), float(y)
        self.color = (255, 255, 0)
        self.move_speed = 0.15

    def update_pos(self):
        dx, dy = self.grid_x - self.world_x, self.grid_y - self.world_y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0.01:
            self.world_x += dx * self.move_speed
            self.world_y += dy * self.move_speed
        else:
            self.world_x, self.world_y = float(self.grid_x), float(self.grid_y)

class Ghost:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.speed = 0.03
        self.color = (180, 220, 255)
        self.trail = []

    def update(self, player_pos, world):
        # Add to trail
        self.trail.append(((self.x, self.y), 20))
        self.trail = [(pos, life - 1) for pos, life in self.trail if life > 0]

        target_x, target_y = player_pos
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            move_x = (dx / dist) * self.speed
            move_y = (dy / dist) * self.speed
            
            # Basic wall avoidance
            next_grid_x, next_grid_y = int(self.x + move_x), int(self.y + move_y)
            if world[next_grid_y][next_grid_x] != 1:
                self.x += move_x
                self.y += move_y

class Puzzle:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.solved = False
        self.highlight_color = (0, 255, 0)
        self.solved_color = (0, 100, 0)

    def interact(self, player):
        pass

    def draw(self, surface, to_iso, offset):
        pass

class LeverPuzzle(Puzzle):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.num_levers = 3
        self.levers = [random.choice([True, False]) for _ in range(self.num_levers)]
        self.solution = [random.choice([True, False]) for _ in range(self.num_levers)]
        self.interaction_target = 0

    def interact(self, player):
        if self.solved: return False
        self.levers[self.interaction_target] = not self.levers[self.interaction_target]
        self.interaction_target = (self.interaction_target + 1) % self.num_levers
        if self.levers == self.solution:
            self.solved = True
            # sfx: puzzle_solve_chime.wav
            return True
        # sfx: lever_click.wav
        return False

    def draw(self, surface, to_iso, offset):
        color = self.solved_color if self.solved else self.highlight_color
        for i in range(self.num_levers):
            px, py = to_iso(self.x + 0.2 + i*0.2, self.y + 0.5)
            pygame.draw.rect(surface, (80,80,80), (px+offset[0]-4, py+offset[1]-15, 8, 20))
            lever_color = (200,200,200) if self.levers[i] else (120,120,120)
            pygame.draw.circle(surface, lever_color, (int(px+offset[0]), int(py+offset[1] - (10 if self.levers[i] else 0))), 3)
        if not self.solved:
            px, py = to_iso(self.x + 0.2 + self.interaction_target*0.2, self.y + 0.5)
            pygame.draw.circle(surface, self.highlight_color, (int(px+offset[0]), int(py+offset[1]+10)), 3, 1)


class PatternPuzzle(Puzzle):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.grid_size = 2
        self.grid = [[random.choice([True, False]) for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def interact(self, player):
        if self.solved: return False
        # Find which panel is closest to player
        px, py = player.grid_x, player.grid_y
        dists = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dists.append(((self.x+i*0.4, self.y+j*0.4), (i,j)))
        
        closest_panel = min(dists, key=lambda item: math.dist((px,py), item[0]))[1]
        ix, iy = closest_panel
        
        # Toggle panels (Lights Out logic)
        for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                self.grid[ny][nx] = not self.grid[ny][nx]
        
        # sfx: panel_press.wav
        if all(all(row) for row in self.grid):
            self.solved = True
            # sfx: puzzle_solve_chime.wav
            return True
        return False

    def draw(self, surface, to_iso, offset):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = self.solved_color if self.solved else (self.highlight_color if self.grid[j][i] else (50,50,50))
                px, py = to_iso(self.x + 0.1 + i*0.4, self.y + 0.1 + j*0.4)
                pygame.draw.rect(surface, color, (px+offset[0]-5, py+offset[1]-5, 10, 10))

class RunePuzzle(Puzzle):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.num_runes = 4
        self.runes = list(range(self.num_runes))
        random.shuffle(self.runes)
        self.solution = self.runes[0]
        self.clue_pos = (0,0) # Set during world gen

    def interact(self, player):
        if self.solved: return False
        # Find which rune is closest to player
        px, py = player.grid_x, player.grid_y
        dists = [(math.dist((px,py), (self.x + (i%2)*0.4, self.y + (i//2)*0.4)), i) for i in range(self.num_runes)]
        closest_rune_idx = min(dists, key=lambda item: item[0])[1]
        
        if self.runes[closest_rune_idx] == self.solution:
            self.solved = True
            # sfx: puzzle_solve_chime.wav
            return True
        # sfx: puzzle_fail_buzz.wav
        return False

    def draw(self, surface, to_iso, offset):
        # Draw clue
        px, py = to_iso(self.clue_pos[0]+0.5, self.clue_pos[1]+0.5)
        self._draw_rune(surface, self.solution, (int(px+offset[0]), int(py+offset[1])), (0,255,255), 10)
        
        # Draw runes
        for i in range(self.num_runes):
            color = self.solved_color if self.solved and self.runes[i] == self.solution else self.highlight_color
            px, py = to_iso(self.x + 0.2 + (i%2)*0.4, self.y + 0.2 + (i//2)*0.4)
            self._draw_rune(surface, self.runes[i], (int(px+offset[0]), int(py+offset[1])), color, 8)

    def _draw_rune(self, surface, idx, pos, color, size):
        points = []
        if idx == 0: # Triangle
            points = [(pos[0], pos[1]-size), (pos[0]-size, pos[1]+size), (pos[0]+size, pos[1]+size)]
        elif idx == 1: # Square
            points = [(pos[0]-size, pos[1]-size), (pos[0]+size, pos[1]-size), (pos[0]+size, pos[1]+size), (pos[0]-size, pos[1]+size)]
        elif idx == 2: # Cross
            pygame.draw.line(surface, color, (pos[0]-size, pos[1]), (pos[0]+size, pos[1]), 2)
            pygame.draw.line(surface, color, (pos[0], pos[1]-size), (pos[0], pos[1]+size), 2)
            return
        elif idx == 3: # Circle
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], size, color)
            return
        if points:
            pygame.gfxdraw.aapolygon(surface, points, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrow keys to move. Space to interact with glowing green puzzles. Escape the house before the ghost gets you or time runs out!"
    game_description = "Escape a procedurally generated haunted house by solving puzzles before a spectral pursuer catches you."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_l = pygame.font.SysFont("monospace", 32)
        
        # Game constants
        self.GRID_W, self.GRID_H = 15, 15
        self.TILE_W, self.TILE_H = 64, 32
        self.MAX_STEPS = 120 * 30 # 120 seconds at 30fps

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_FLOOR = (30, 30, 45)
        self.COLOR_WALL = (40, 40, 55)
        self.COLOR_WALL_TOP = (60, 60, 80)
        self.COLOR_DOOR = (139, 69, 19)
        self.COLOR_DOOR_OPEN = (0, 0, 0)
        
        # Initialize state variables
        self.player = None
        self.ghost = None
        self.world = None
        self.puzzles = []
        self.exit_pos = (0,0)
        self.exit_open = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.dust_motes = []

        self.reset()
        self.validate_implementation()

    def _generate_world(self):
        # 0: empty, 1: wall, 2: floor
        self.world = np.ones((self.GRID_H, self.GRID_W), dtype=int)
        
        # Simple room generation
        rooms = []
        for _ in range(10): # Attempt to place 10 rooms
            w, h = random.randint(3, 5), random.randint(3, 5)
            x, y = random.randint(1, self.GRID_W-w-1), random.randint(1, self.GRID_H-h-1)
            new_room = pygame.Rect(x, y, w, h)
            if not any(new_room.colliderect(r) for r in rooms):
                rooms.append(new_room)
        
        # Carve rooms and corridors
        for r1 in rooms:
            self.world[r1.top:r1.bottom, r1.left:r1.right] = 2
            # Connect to another random room
            r2 = random.choice(rooms)
            x1, y1 = r1.center
            x2, y2 = r2.center
            for x in range(min(x1,x2), max(x1,x2)+1): self.world[y1, x] = 2
            for y in range(min(y1,y2), max(y1,y2)+1): self.world[y, x2] = 2

        floor_tiles = list(zip(*np.where(self.world == 2)))
        if not floor_tiles: # Failsafe
            self.world[self.GRID_H//2, self.GRID_W//2] = 2
            floor_tiles = [(self.GRID_W//2, self.GRID_H//2)]
        
        # Place player
        start_pos = random.choice(floor_tiles)
        self.player = Player(start_pos[1], start_pos[0])
        
        # Place exit
        self.exit_pos = random.choice(floor_tiles)
        while math.dist(start_pos, self.exit_pos) < 5:
            self.exit_pos = random.choice(floor_tiles)
        self.exit_pos = (self.exit_pos[1], self.exit_pos[0])

        # Place puzzles
        self.puzzles = []
        puzzle_types = [LeverPuzzle, PatternPuzzle, RunePuzzle]
        puzzle_locs = []
        while len(self.puzzles) < 3:
            pos = random.choice(floor_tiles)
            if pos != start_pos and pos != self.exit_pos and pos not in puzzle_locs:
                puzzle_type = puzzle_types[len(self.puzzles)]
                self.puzzles.append(puzzle_type(pos[1], pos[0]))
                puzzle_locs.append(pos)
        
        # Special setup for rune puzzle clue
        rune_puzzle = next(p for p in self.puzzles if isinstance(p, RunePuzzle))
        clue_pos = random.choice(floor_tiles)
        while math.dist(clue_pos, (rune_puzzle.x, rune_puzzle.y)) < 4 or clue_pos == start_pos:
            clue_pos = random.choice(floor_tiles)
        rune_puzzle.clue_pos = (clue_pos[1], clue_pos[0])
        
        # Place ghost
        ghost_pos = random.choice(floor_tiles)
        while math.dist(ghost_pos, start_pos) < 6:
            ghost_pos = random.choice(floor_tiles)
        self.ghost = Ghost(ghost_pos[1], ghost_pos[0])
        
        # Generate dust motes
        self.dust_motes = [(random.randint(0, self.width), random.randint(0, self.height), random.uniform(0.1, 0.5), random.randint(1,2)) for _ in range(100)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_world()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.exit_open = False
        self.prev_space_held = False
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        # --- Player Movement ---
        target_x, target_y = self.player.grid_x, self.player.grid_y
        if movement == 1: target_y -= 1 # Up
        elif movement == 2: target_y += 1 # Down
        elif movement == 3: target_x -= 1 # Left
        elif movement == 4: target_x += 1 # Right

        if 0 <= target_x < self.GRID_W and 0 <= target_y < self.GRID_H and self.world[target_y, target_x] == 2:
            self.player.grid_x, self.player.grid_y = target_x, target_y
            # sfx: player_footstep.wav
        
        # --- Player Interaction ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            for puzzle in self.puzzles:
                if not puzzle.solved and math.dist((self.player.grid_x, self.player.grid_y), (puzzle.x, puzzle.y)) < 1.5:
                    if puzzle.interact(self.player):
                        reward += 5.0
                        self.score += 5
                    break # Interact with one puzzle at a time
        self.prev_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        self.player.update_pos()
        self.ghost.update((self.player.world_x, self.player.world_y), self.world)
        
        # Increase ghost speed over time
        if self.steps % (30 * 30) == 0:
            self.ghost.speed += 0.005

        # Check if all puzzles are solved
        if not self.exit_open and all(p.solved for p in self.puzzles):
            self.exit_open = True
            # sfx: door_unlock.wav
            self.score += 10 # Bonus for opening door
            reward += 10.0

        # --- Termination Conditions ---
        terminated = False
        if math.dist((self.player.world_x, self.player.world_y), (self.ghost.x, self.ghost.y)) < 0.8:
            terminated = True
            reward = -100.0
            self.score -= 100
            # sfx: player_caught.wav
        
        if self.exit_open and (self.player.grid_x, self.player.grid_y) == self.exit_pos:
            terminated = True
            reward = 100.0
            self.score += 100
            # sfx: victory_fanfare.wav

        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50.0
            self.score -= 50
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _to_iso(self, x, y):
        iso_x = (x - y) * (self.TILE_W / 2)
        iso_y = (x + y) * (self.TILE_H / 2)
        return iso_x, iso_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Update and draw dust motes
        for i, (x, y, speed, size) in enumerate(self.dust_motes):
            y = (y + speed) % self.height
            self.dust_motes[i] = (x, y, speed, size)
            pygame.draw.circle(self.screen, (50,50,60), (int(x), int(y)), size)

        offset_x = self.width / 2
        offset_y = self.height / 4

        # --- Render World ---
        entities_to_render = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                tile = self.world[y, x]
                if tile == 2: # Floor
                    iso_x, iso_y = self._to_iso(x, y)
                    points = [
                        (iso_x + offset_x, iso_y + offset_y),
                        (iso_x + self.TILE_W / 2 + offset_x, iso_y + self.TILE_H / 2 + offset_y),
                        (iso_x + offset_x, iso_y + self.TILE_H + offset_y),
                        (iso_x - self.TILE_W / 2 + offset_x, iso_y + self.TILE_H / 2 + offset_y),
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_FLOOR, points)
                    
                    # Add entities in this tile to render list
                    if (x,y) == self.exit_pos:
                        entities_to_render.append(('exit', y, x, y))
                    for p in self.puzzles:
                        if (p.x, p.y) == (x,y):
                             entities_to_render.append(('puzzle', p, x, y))
        
        # Add dynamic entities
        entities_to_render.append(('ghost', self.ghost, self.ghost.x, self.ghost.y))
        entities_to_render.append(('player', self.player, self.player.world_x, self.player.world_y))

        # Sort entities by Y-coordinate for proper occlusion
        entities_to_render.sort(key=lambda e: e[3])

        # --- Render Entities ---
        for entity_type, obj, _, _ in entities_to_render:
            if entity_type == 'exit':
                px, py = self._to_iso(self.exit_pos[0], self.exit_pos[1])
                color = self.COLOR_DOOR_OPEN if self.exit_open else self.COLOR_DOOR
                rect = pygame.Rect(px+offset_x-16, py+offset_y-32, 32, 48)
                pygame.draw.rect(self.screen, color, rect)
                if self.exit_open:
                    pygame.draw.rect(self.screen, (255,255,255), rect, 2)
            elif entity_type == 'puzzle':
                obj.draw(self.screen, self._to_iso, (offset_x, offset_y))
            elif entity_type == 'ghost':
                # Trail
                for (tx, ty), life in obj.trail:
                    alpha = int((life/20) * 100)
                    color = (self.ghost.color[0], self.ghost.color[1], self.ghost.color[2], alpha)
                    s = pygame.Surface((20, 30), pygame.SRCALPHA)
                    px, py = self._to_iso(tx, ty)
                    pygame.draw.circle(s, color, (10, 15), int(8 * (life/20)))
                    self.screen.blit(s, (px+offset_x-10, py+offset_y-22))
                # Ghost body
                px, py = self._to_iso(obj.x, obj.y)
                s = pygame.Surface((32, 48), pygame.SRCALPHA)
                pygame.gfxdraw.aaellipse(s, 16, 24, 12, 20, self.ghost.color)
                pygame.gfxdraw.filled_ellipse(s, 16, 24, 12, 20, self.ghost.color)
                self.screen.blit(s, (px+offset_x-16, py+offset_y-40))
            elif entity_type == 'player':
                px, py = self._to_iso(obj.world_x, obj.world_y)
                pygame.draw.circle(self.screen, (0,0,0), (int(px+offset_x), int(py+offset_y+5)), 10)
                pygame.draw.circle(self.screen, obj.color, (int(px+offset_x), int(py+offset_y)), 10)
        
        # --- Render Walls (after entities for occlusion) ---
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.world[y,x] == 1 and any(self.world[ny,nx]==2 for nx,ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)] if 0<=nx<self.GRID_W and 0<=ny<self.GRID_H):
                    px, py = self._to_iso(x, y)
                    top_points = [(px+offset_x, py+offset_y), (px+self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y), (px+offset_x, py+self.TILE_H+offset_y), (px-self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y)]
                    left_points = [(px-self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y), (px-self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y-64), (px+offset_x, py+offset_y-64), (px+offset_x, py+offset_y)]
                    right_points = [(px+self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y), (px+self.TILE_W/2+offset_x, py+self.TILE_H/2+offset_y-64), (px+offset_x, py+offset_y-64), (px+offset_x, py+offset_y)]
                    pygame.draw.polygon(self.screen, self.COLOR_WALL, left_points)
                    pygame.draw.polygon(self.screen, self.COLOR_WALL, right_points)
                    pygame.draw.polygon(self.screen, self.COLOR_WALL_TOP, top_points)

        # --- Render UI & Effects ---
        # Ghost proximity vignette
        dist_to_ghost = math.dist((self.player.world_x, self.player.world_y), (self.ghost.x, self.ghost.y))
        if dist_to_ghost < 5:
            vignette_alpha = int(max(0, min(200, (1 - dist_to_ghost / 5) * 255)))
            vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.ellipse(vignette, (0,0,0,vignette_alpha), (0,0,self.width, self.height))
            pygame.draw.ellipse(vignette, (0,0,0,0), (50,50,self.width-100, self.height-100))
            self.screen.blit(vignette, (0,0))
            # sfx: heartbeat_fast.wav (if dist < 2)

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30
        timer_text = self.font_s.render(f"TIME: {time_left:.1f}", True, (255, 255, 255))
        self.screen.blit(timer_text, (10, 10))
        
        # Puzzle status
        puzzles_solved_count = sum(1 for p in self.puzzles if p.solved)
        puzzle_text = self.font_s.render(f"PUZZLES: {puzzles_solved_count}/{len(self.puzzles)}", True, (255, 255, 255))
        self.screen.blit(puzzle_text, (10, 30))

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU ESCAPED!" if self.score > 0 else "YOU WERE CAUGHT"
            end_text = self.font_l.render(msg, True, (255,50,50))
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "puzzles_solved": sum(1 for p in self.puzzles if p.solved),
            "time_left": (self.MAX_STEPS - self.steps) / 30
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Haunted House Escape")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()