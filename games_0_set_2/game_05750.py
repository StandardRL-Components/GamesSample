import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press Space to interact with objects. "
        "Avoid the red ghosts!"
    )

    game_description = (
        "Escape a haunted house by finding the key and reaching the exit. "
        "Solve a puzzle to access the key, but beware of patrolling ghosts and a ticking clock."
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
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # --- Game Constants ---
        self.FPS = 30
        self.GRID_SIZE = 4
        self.MAX_TIME = 60 * self.FPS  # 60 seconds
        self.MAX_CAPTURES = 3
        
        # --- Colors ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_FLOOR = (25, 35, 45)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_GHOST = (255, 50, 50)
        self.COLOR_INTERACT = (255, 220, 0)
        self.COLOR_EXIT = (50, 255, 50)
        self.COLOR_LOCKED = (200, 0, 0)
        self.COLOR_TEXT = (230, 230, 230)
        
        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)

        # --- Room rendering geometry ---
        self.room_center = pygame.math.Vector2(self.screen_size[0] // 2, self.screen_size[1] // 2 + 30)
        self.room_width = 500
        self.room_height = 280
        self.wall_height = 80

        # --- State variables will be initialized in reset() ---
        # self.reset() is called by the API, so we don't call it here.
        
    def _generate_map(self):
        grid = [[{'visited': False, 'doors': {'N': False, 'S': False, 'E': False, 'W': False}} for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        stack = deque()
        start_cell = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
        grid[start_cell[0]][start_cell[1]]['visited'] = True
        stack.append(start_cell)
        
        dead_ends = []
        
        while stack:
            r, c = stack[-1]
            neighbors = []
            if r > 0 and not grid[r-1][c]['visited']: neighbors.append(('N', r-1, c))
            if r < self.GRID_SIZE - 1 and not grid[r+1][c]['visited']: neighbors.append(('S', r+1, c))
            if c > 0 and not grid[r][c-1]['visited']: neighbors.append(('W', r, c-1))
            if c < self.GRID_SIZE - 1 and not grid[r][c+1]['visited']: neighbors.append(('E', r, c+1))
            
            if neighbors:
                # FIX: np.random.choice does not correctly handle a list of tuples with mixed types.
                # It converts everything to strings, causing a TypeError when using string-cast numbers as indices.
                # The correct way is to pick a random index and use it to select from the Python list.
                chosen_neighbor = neighbors[self.np_random.integers(len(neighbors))]
                direction, nr, nc = chosen_neighbor
                
                if direction == 'N':
                    grid[r][c]['doors']['N'] = True
                    grid[nr][nc]['doors']['S'] = True
                elif direction == 'S':
                    grid[r][c]['doors']['S'] = True
                    grid[nr][nc]['doors']['N'] = True
                elif direction == 'W':
                    grid[r][c]['doors']['W'] = True
                    grid[nr][nc]['doors']['E'] = True
                elif direction == 'E':
                    grid[r][c]['doors']['E'] = True
                    grid[nr][nc]['doors']['W'] = True
                
                grid[nr][nc]['visited'] = True
                stack.append((nr, nc))
            else:
                popped = stack.pop()
                # Check if it's a dead end (only one door)
                if sum(grid[popped[0]][popped[1]]['doors'].values()) == 1:
                    dead_ends.append(popped)

        if len(dead_ends) < 3: # Ensure enough unique rooms for objectives
            return self._generate_map()

        return grid, dead_ends

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.timer = self.MAX_TIME
        self.captures = 0
        
        # --- Map and Entities ---
        self.map, dead_ends = self._generate_map()
        self.np_random.shuffle(dead_ends)
        
        self.player_room = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        
        # Ensure objective rooms are not the start room
        valid_dead_ends = [de for de in dead_ends if de != self.player_room]
        if len(valid_dead_ends) < 3: # Regenerate if layout is too cramped
            return self.reset(seed=seed, options=options)

        self.lever_room = valid_dead_ends.pop()
        self.key_room = valid_dead_ends.pop()
        self.exit_room = valid_dead_ends.pop()

        self.lever_pos = pygame.math.Vector2(0, 0.7) # Relative to room center
        self.key_pos = pygame.math.Vector2(0, -0.7)
        
        self.lever_flipped = False
        self.has_key = False
        
        # Lock the key room
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                for d, (dr, dc) in {'N': (-1,0), 'S': (1,0), 'W': (0,-1), 'E': (0,1)}.items():
                    if (r+dr, c+dc) == self.key_room and self.map[r][c]['doors'][d]:
                        self.map[r][c]['doors'][d] = 'locked'
                        # Find corresponding door and lock it too
                        if d == 'N': self.map[r-1][c]['doors']['S'] = 'locked'
                        if d == 'S': self.map[r+1][c]['doors']['N'] = 'locked'
                        if d == 'W': self.map[r][c-1]['doors']['E'] = 'locked'
                        if d == 'E': self.map[r][c+1]['doors']['W'] = 'locked'

        # --- Player ---
        self.player_pos = pygame.math.Vector2(0, 0) # Relative to room center
        self.player_speed = 0.03
        self.player_radius = 12

        # --- Ghosts ---
        self.ghosts = []
        num_ghosts = 2
        possible_ghost_rooms = [ (r,c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE) if (r,c) not in [self.player_room, self.key_room, self.exit_room, self.lever_room]]
        self.np_random.shuffle(possible_ghost_rooms)
        for i in range(min(num_ghosts, len(possible_ghost_rooms))):
            room = possible_ghost_rooms[i]
            self.ghosts.append({
                'room': room,
                'pos': pygame.math.Vector2(self.np_random.uniform(-0.8, 0.8), self.np_random.uniform(-0.8, 0.8)),
                'path': [pygame.math.Vector2(self.np_random.uniform(-0.8, 0.8), self.np_random.uniform(-0.8, 0.8)) for _ in range(4)],
                'target_idx': 0,
                'speed': 0.015,
                'radius': 15
            })

        # --- Misc State ---
        self.interaction_cooldown = 0
        self.particles = []
        self.visited_rooms = {self.player_room}
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def _rel_to_abs_pos(self, rel_pos):
        x = self.room_center.x + rel_pos.x * (self.room_width / 2)
        y = self.room_center.y + rel_pos.y * (self.room_height / 2) * 0.7 # Isometric perspective adjustment
        return pygame.math.Vector2(x, y)

    def step(self, action):
        self.steps += 1
        reward = -0.01  # Time penalty
        
        self.timer -= 1
        terminated = self.timer <= 0
        truncated = False # No truncation condition in this game

        if terminated:
            if not self.win:
                reward -= 100
            return self._get_observation(), reward, terminated, truncated, self._get_info()

        # --- Update Cooldowns ---
        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Unpack Action ---
        movement, space_held, _ = action
        
        # --- Player Movement ---
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.player_speed

        # Clamp player position to room boundaries
        self.player_pos.x = max(-1, min(1, self.player_pos.x))
        self.player_pos.y = max(-1, min(1, self.player_pos.y))

        # --- Room Transitions ---
        current_doors = self.map[self.player_room[0]][self.player_room[1]]['doors']
        new_room = None
        if self.player_pos.y < -0.95 and current_doors['N'] is True:
            new_room = (self.player_room[0] - 1, self.player_room[1])
            self.player_pos.y = 0.95
        elif self.player_pos.y > 0.95 and current_doors['S'] is True:
            new_room = (self.player_room[0] + 1, self.player_room[1])
            self.player_pos.y = -0.95
        elif self.player_pos.x < -0.95 and current_doors['W'] is True:
            new_room = (self.player_room[0], self.player_room[1] - 1)
            self.player_pos.x = 0.95
        elif self.player_pos.x > 0.95 and current_doors['E'] is True:
            new_room = (self.player_room[0], self.player_room[1] + 1)
            self.player_pos.x = -0.95
        
        if new_room:
            self.player_room = new_room
            if new_room not in self.visited_rooms:
                self.visited_rooms.add(new_room)
                reward += 0.1

        # --- Player Interaction ---
        if space_held and self.interaction_cooldown == 0:
            self.interaction_cooldown = self.FPS // 2 # 0.5 sec cooldown
            
            # Interact with lever
            if self.player_room == self.lever_room and not self.lever_flipped:
                if self._rel_to_abs_pos(self.player_pos).distance_to(self._rel_to_abs_pos(self.lever_pos)) < 40:
                    self.lever_flipped = True
                    reward += 5.0
                    # Unlock the key room
                    for r in range(self.GRID_SIZE):
                        for c in range(self.GRID_SIZE):
                            for d in ['N', 'S', 'W', 'E']:
                                if self.map[r][c]['doors'][d] == 'locked':
                                    self.map[r][c]['doors'][d] = True
                    self._spawn_particles(self._rel_to_abs_pos(self.lever_pos), self.COLOR_INTERACT, 20)

            # Interact with key
            if self.player_room == self.key_room and not self.has_key and self.lever_flipped:
                if self._rel_to_abs_pos(self.player_pos).distance_to(self._rel_to_abs_pos(self.key_pos)) < 40:
                    self.has_key = True
                    reward += 10.0
                    self._spawn_particles(self._rel_to_abs_pos(self.key_pos), self.COLOR_INTERACT, 30)
            
            # Interact with exit
            if self.player_room == self.exit_room and self.has_key:
                if self.player_pos.y < -0.8: # Exit is at the north wall
                    terminated = True
                    self.win = True
                    reward += 100.0

        # --- Update Ghosts ---
        base_speed_increase = self.steps / (10 * self.FPS) * 0.0005
        for ghost in self.ghosts:
            ghost['speed'] = 0.015 + base_speed_increase
            target_pos = ghost['path'][ghost['target_idx']]
            direction = (target_pos - ghost['pos'])
            if direction.length() < 0.1:
                ghost['target_idx'] = (ghost['target_idx'] + 1) % len(ghost['path'])
            else:
                ghost['pos'] += direction.normalize() * ghost['speed']
            
            # Check for ghost capture
            if ghost['room'] == self.player_room:
                abs_ghost_pos = self._rel_to_abs_pos(ghost['pos'])
                abs_player_pos = self._rel_to_abs_pos(self.player_pos)
                if abs_ghost_pos.distance_to(abs_player_pos) < ghost['radius'] + self.player_radius:
                    self.captures += 1
                    reward -= 10.0
                    self.player_pos = pygame.math.Vector2(0, 0.8) # Move to room entrance
                    self.screen_shake = 10
                    if self.captures >= self.MAX_CAPTURES:
                        terminated = True
                        reward -= 100.0

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        offset = pygame.math.Vector2(0,0)
        if self.screen_shake > 0:
            offset.x = self.np_random.integers(-5, 6)
            offset.y = self.np_random.integers(-5, 6)

        self.screen.fill(self.COLOR_BG)
        
        self._render_room(offset)
        self._render_entities(offset)
        self._render_particles(offset)
        self._render_lighting(offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_room(self, offset):
        # Floor
        floor_points = [
            self.room_center + offset + pygame.math.Vector2(0, -self.room_height / 2),
            self.room_center + offset + pygame.math.Vector2(self.room_width / 2, 0),
            self.room_center + offset + pygame.math.Vector2(0, self.room_height / 2),
            self.room_center + offset + pygame.math.Vector2(-self.room_width / 2, 0)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in floor_points], self.COLOR_FLOOR)
        
        # Walls
        doors = self.map[self.player_room[0]][self.player_room[1]]['doors']
        door_width_ratio = 0.3
        
        # Far walls (rendered first)
        # North-West wall
        if not doors['N'] and not doors['W']:
             pygame.draw.polygon(self.screen, self.COLOR_WALL, [floor_points[3], floor_points[0], floor_points[0] - (0, self.wall_height), floor_points[3] - (0, self.wall_height)])
        # North-East wall
        if not doors['N'] and not doors['E']:
            pygame.draw.polygon(self.screen, self.COLOR_WALL, [floor_points[0], floor_points[1], floor_points[1] - (0, self.wall_height), floor_points[0] - (0, self.wall_height)])

        # Doorways
        door_color = self.COLOR_EXIT if self.player_room == self.exit_room else self.COLOR_FLOOR
        if doors['N'] == 'locked': door_color = self.COLOR_LOCKED
        
        # North doorway
        p1 = floor_points[3].lerp(floor_points[0], 0.5 - door_width_ratio/2)
        p2 = floor_points[3].lerp(floor_points[0], 0.5 + door_width_ratio/2)
        pygame.draw.polygon(self.screen, self.COLOR_WALL, [floor_points[3], p1, p1 - (0, self.wall_height), floor_points[3] - (0, self.wall_height)])
        pygame.draw.polygon(self.screen, self.COLOR_WALL, [p2, floor_points[0], floor_points[0] - (0, self.wall_height), p2 - (0, self.wall_height)])
        if doors['N']: pygame.draw.line(self.screen, door_color, p1, p2, 5)

    def _render_entities(self, offset):
        # --- Render objects in depth order (back to front) ---
        entities = []

        # Lever
        if self.player_room == self.lever_room:
            entities.append({'pos': self.lever_pos, 'type': 'lever', 'state': self.lever_flipped})
        # Key
        if self.player_room == self.key_room and not self.has_key and self.lever_flipped:
            entities.append({'pos': self.key_pos, 'type': 'key'})
        # Ghosts
        for ghost in self.ghosts:
            if ghost['room'] == self.player_room:
                entities.append({'pos': ghost['pos'], 'type': 'ghost', 'radius': ghost['radius']})
        # Player
        entities.append({'pos': self.player_pos, 'type': 'player', 'radius': self.player_radius})
        
        entities.sort(key=lambda e: e['pos'].y)

        for entity in entities:
            abs_pos = self._rel_to_abs_pos(entity['pos']) + offset
            if entity['type'] == 'player':
                pygame.gfxdraw.filled_circle(self.screen, int(abs_pos.x), int(abs_pos.y), int(entity['radius']), self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, int(abs_pos.x), int(abs_pos.y), int(entity['radius']), self.COLOR_PLAYER)
            elif entity['type'] == 'ghost':
                # Pulsating effect
                pulse = math.sin(self.steps * 0.2) * 3
                color = (*self.COLOR_GHOST, 100 + int(pulse * 10))
                radius = entity['radius'] + pulse
                
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius), color)
                self.screen.blit(temp_surf, (int(abs_pos.x - radius), int(abs_pos.y - radius)))
            elif entity['type'] == 'lever':
                lever_base = abs_pos + (0, 5)
                lever_top = abs_pos - (0, 20)
                handle_dir = -15 if entity['state'] else 15
                handle_pos = lever_top + (handle_dir, 0)
                pygame.draw.rect(self.screen, self.COLOR_INTERACT, (lever_base.x - 10, lever_base.y, 20, 5))
                pygame.draw.line(self.screen, (150,150,150), lever_base, handle_pos, 4)
                pygame.gfxdraw.filled_circle(self.screen, int(handle_pos.x), int(handle_pos.y), 6, self.COLOR_INTERACT)
            elif entity['type'] == 'key':
                p = abs_pos
                pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y-5), 7, self.COLOR_INTERACT)
                pygame.draw.rect(self.screen, self.COLOR_INTERACT, (p.x-3, p.y, 6, 15))
                pygame.draw.rect(self.screen, self.COLOR_INTERACT, (p.x-3, p.y+10, 12, 5))

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p['pos'] + offset
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
            self.screen.blit(temp_surf, (int(pos.x - p['radius']), int(pos.y - p['radius'])))

    def _render_lighting(self, offset):
        light_surf = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        light_surf.fill((0, 0, 0, 210))
        
        player_abs_pos = self._rel_to_abs_pos(self.player_pos) + offset
        
        # Player light
        radius = 120 + math.sin(self.steps * 0.1) * 5
        pygame.draw.circle(light_surf, (0,0,0,0), (int(player_abs_pos.x), int(player_abs_pos.y)), int(radius))

        # Other light sources
        if self.player_room == self.lever_room:
             pygame.draw.circle(light_surf, (0,0,0,150), (self._rel_to_abs_pos(self.lever_pos) + offset), 30)
        if self.player_room == self.key_room and self.lever_flipped and not self.has_key:
             pygame.draw.circle(light_surf, (0,0,0,100), (self._rel_to_abs_pos(self.key_pos) + offset), 40)
        if self.player_room == self.exit_room:
             p1 = self.room_center.lerp(self.room_center + pygame.math.Vector2(-self.room_width/2, -self.room_height/2), 0.5) + offset
             pygame.draw.circle(light_surf, (0,0,0,150), p1, 50)
        
        self.screen.blit(light_surf, (0,0))

    def _render_ui(self):
        # Timer
        time_text = f"Time: {max(0, self.timer // self.FPS):02d}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.screen_size[0] - time_surf.get_width() - 10, 10))
        
        # Captures
        for i in range(self.MAX_CAPTURES):
            color = self.COLOR_GHOST if i < self.captures else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, 25 + i * 30, 25, 10, color)
            pygame.gfxdraw.aacircle(self.screen, 25 + i * 30, 25, 10, color)
            
        # Key
        if self.has_key:
            p = pygame.math.Vector2(25 + self.MAX_CAPTURES * 30, 25)
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y-3), 5, self.COLOR_INTERACT)
            pygame.draw.rect(self.screen, self.COLOR_INTERACT, (p.x-2, p.y, 4, 10))
            pygame.draw.rect(self.screen, self.COLOR_INTERACT, (p.x-2, p.y+6, 8, 3))
        
        # Game Over / Win Text
        if self.timer <= 0 or (self.captures >= self.MAX_CAPTURES) or self.win:
            s = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            text = "YOU ESCAPED!" if self.win else "GAME OVER"
            color = self.COLOR_EXIT if self.win else self.COLOR_GHOST
            
            text_surf = self.font_main.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_size[0]/2, self.screen_size[1]/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "captures": self.captures,
            "has_key": self.has_key,
            "player_room": self.player_room,
            "win": self.win,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # The environment can be run headless, but for human play, we need a visible display.
    # We'll unset the dummy video driver if it was set.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Haunted House Escape")
    screen = pygame.display.set_mode(env.screen_size)
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    
    print(env.user_guide)
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        # This event loop is crucial for a responsive window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Win: {info['win']}, Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Let the game over screen show for a bit before auto-resetting
            # In a real training loop, you would just call reset immediately.
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # --- Pygame Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()