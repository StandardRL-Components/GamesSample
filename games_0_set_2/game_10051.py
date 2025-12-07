import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:50:47.281535
# Source Brief: brief_00051.md
# Brief Index: 51
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, size, lifetime, vx, vy, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.vx = vx
        self.vy = vy
        self.gravity = gravity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            color = (*self.color, alpha)
            center = (int(self.x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(self.size), color)
            pygame.gfxdraw.aacircle(surface, center[0], center[1], int(self.size), color)

class Room:
    """Represents a single room in the labyrinth."""
    def __init__(self, grid_size, index):
        self.index = index
        self.grid_size = grid_size
        self.cell_size = 40
        self.walls = [set(), set(), set()]  # Past, Present, Future
        self.materials = set()
        self.door_pos = None
        self.door_target = None
        self.exit_pos = None
        self.start_pos = (1, 1)
        self.crafting_station = (1, grid_size[1] - 2)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting labyrinth by manipulating time. "
        "Collect dream materials to craft keys and find the exit."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to interact with objects. "
        "Press shift to cycle through time (Past, Present, Future)."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 2000
    NUM_ROOMS = 5

    # --- COLORS ---
    COLOR_BG = (10, 20, 35)
    COLOR_PAST = (50, 100, 220)
    COLOR_PRESENT = (50, 220, 100)
    COLOR_FUTURE = (220, 50, 100)
    TIME_COLORS = [COLOR_PAST, COLOR_PRESENT, COLOR_FUTURE]
    TIME_NAMES = ["PAST", "PRESENT", "FUTURE"]

    COLOR_PLAYER = (200, 255, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_MATERIAL = (255, 215, 0)
    COLOR_KEY = (192, 192, 192)
    COLOR_DOOR = (150, 75, 0)
    COLOR_EXIT = (255, 0, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_WALL_HINT = (40, 50, 70)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_time = pygame.font.SysFont("monospace", 24, bold=True)

        self.rooms = []
        self.particles = []
        
        self.player_grid_pos = np.array([0, 0])
        self.player_render_pos = np.array([0.0, 0.0])

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.time_state = 1  # 0: Past, 1: Present, 2: Future
        self.dream_materials = 0
        self.keys = 0
        self.level = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles.clear()
        
        self._generate_labyrinth()
        self._load_room(0)
        
        self.player_render_pos = self.player_grid_pos * self.CELL_SIZE + self.CELL_SIZE / 2

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Input and Update State ---
        if shift_pressed:
            self.time_state = (self.time_state + 1) % 3
            self._create_time_shift_effect()
            # SFX: TimeWarp

        self._handle_movement(movement)

        if space_pressed:
            reward += self._handle_interaction()

        # --- Update Game World ---
        self._update_player_render_pos()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_labyrinth(self):
        self.rooms.clear()
        for i in range(self.NUM_ROOMS):
            room = Room((self.GRID_W, self.GRID_H), i)
            self._generate_room_layout(room)
            self.rooms.append(room)

    def _generate_room_layout(self, room):
        # Generate a basic maze using recursive backtracking
        stack = [(room.start_pos)]
        visited = {room.start_pos}
        
        # Present-day path is always clear
        present_path = set()
        present_path.add(room.start_pos)
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 1 <= nx < self.GRID_W - 1 and 1 <= ny < self.GRID_H - 1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                wall_x, wall_y = cx + (nx - cx) // 2, cy + (ny - cy) // 2
                present_path.add((wall_x, wall_y))
                present_path.add((nx, ny))
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Create walls for all time states, then remove path for present
        for t in range(3):
            for x in range(self.GRID_W):
                for y in range(self.GRID_H):
                    if x == 0 or y == 0 or x == self.GRID_W - 1 or y == self.GRID_H - 1 or (x % 2 == 0 or y % 2 == 0):
                        room.walls[t].add((x, y))
        
        for pos in present_path:
            room.walls[1].discard(pos)

        # Add time-specific walls/paths for Past and Future
        for pos in present_path:
            if random.random() < 0.3 + (room.index * 0.1): # More complex over time
                if random.random() < 0.5:
                    room.walls[0].add(pos) # Block in past
                else:
                    room.walls[2].add(pos) # Block in future
        
        # Ensure start and end are always clear
        for t in range(3):
            room.walls[t].discard(room.start_pos)
            room.walls[t].discard(room.crafting_station)

        # Place materials
        for _ in range(3 + room.index):
            while True:
                x, y = random.randint(1, self.GRID_W - 2), random.randint(1, self.GRID_H - 2)
                if (x, y) not in room.walls[0] or (x, y) not in room.walls[1] or (x, y) not in room.walls[2]:
                    room.materials.add((x, y))
                    break

        # Place door or exit
        if room.index < self.NUM_ROOMS - 1:
            room.door_pos = (self.GRID_W - 2, random.randint(1, self.GRID_H - 2))
            room.door_target = room.index + 1
            for t in range(3):
                room.walls[t].discard(room.door_pos)
        else:
            room.exit_pos = (self.GRID_W - 2, self.GRID_H // 2)
            for t in range(3):
                room.walls[t].discard(room.exit_pos)

    def _load_room(self, room_index):
        self.level = room_index
        room = self.rooms[room_index]
        self.player_grid_pos = np.array(room.start_pos)
        self.current_room = room

    def _handle_movement(self, movement):
        if movement == 0: return # No-op
        
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]
        
        target_pos = (self.player_grid_pos[0] + dx, self.player_grid_pos[1] + dy)
        
        if target_pos not in self.current_room.walls[self.time_state]:
            self.player_grid_pos = np.array(target_pos)

    def _handle_interaction(self):
        pos_tuple = tuple(self.player_grid_pos)
        reward = 0

        # Collect material
        if pos_tuple in self.current_room.materials:
            self.current_room.materials.remove(pos_tuple)
            self.dream_materials += 1
            self.score += 1
            reward += 1
            self._create_collect_effect(pos_tuple)
            # SFX: Collect
            return reward

        # Craft key
        if pos_tuple == self.current_room.crafting_station:
            cost = 3 + (self.level // 3)
            if self.dream_materials >= cost:
                self.dream_materials -= cost
                self.keys += 1
                self.score += 5
                reward += 5
                self._create_crafting_effect(pos_tuple)
                # SFX: CraftKey
                return reward

        # Use key on door
        if self.current_room.door_pos and pos_tuple == self.current_room.door_pos and self.keys > 0:
            self.keys -= 1
            self.score += 10
            reward += 10
            self._load_room(self.current_room.door_target)
            self._create_unlock_effect(pos_tuple)
            # SFX: UnlockDoor
            return reward

        # Reach exit
        if self.current_room.exit_pos and pos_tuple == self.current_room.exit_pos:
            self.score += 100
            reward += 100
            self.game_over = True
            # SFX: Victory
            return reward
        
        return reward

    def _update_player_render_pos(self):
        target_pos = self.player_grid_pos * self.CELL_SIZE + self.CELL_SIZE / 2
        self.player_render_pos = self.player_render_pos * 0.6 + target_pos * 0.4

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

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
            "level": self.level,
            "keys": self.keys,
            "materials": self.dream_materials,
        }

    def _render_game(self):
        self._render_room()
        self._render_player()
        self._render_particles()

    def _render_room(self):
        # Draw faint hints of walls from other time states
        for t in range(3):
            if t != self.time_state:
                for x, y in self.current_room.walls[t]:
                    if (x,y) not in self.current_room.walls[self.time_state]:
                        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, self.COLOR_WALL_HINT, rect, 1)

        # Draw current walls
        wall_color = self.TIME_COLORS[self.time_state]
        for x, y in self.current_room.walls[self.time_state]:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, wall_color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Border for definition

        # Draw materials
        for x, y in self.current_room.materials:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            size = int(self.CELL_SIZE * 0.2 + pulse * 3)
            pos = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_MATERIAL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_MATERIAL)

        # Draw crafting station
        x, y = self.current_room.crafting_station
        pos = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
        r = self.CELL_SIZE * 0.3
        points = []
        for i in range(6):
            angle = math.radians(60 * i + self.steps * 2)
            points.append((pos[0] + r * math.cos(angle), pos[1] + r * math.sin(angle)))
        pygame.draw.aalines(self.screen, self.COLOR_KEY, True, points)
        
        # Draw door
        if self.current_room.door_pos:
            x, y = self.current_room.door_pos
            rect = pygame.Rect(x * self.CELL_SIZE + 5, y * self.CELL_SIZE + 5, self.CELL_SIZE - 10, self.CELL_SIZE - 10)
            pygame.draw.rect(self.screen, self.COLOR_DOOR, rect)
            pygame.draw.circle(self.screen, self.COLOR_KEY, rect.center, 5)

        # Draw exit
        if self.current_room.exit_pos:
            x, y = self.current_room.exit_pos
            pos = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
            for i in range(5):
                alpha = 100 + i * 30
                radius = int(self.CELL_SIZE * 0.4 - i * 2 + math.sin(self.steps * 0.1 + i) * 2)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_EXIT, alpha))

    def _render_player(self):
        pos = (int(self.player_render_pos[0]), int(self.player_render_pos[1]))
        radius = int(self.CELL_SIZE * 0.35)
        
        # Glow
        glow_radius = int(radius * 1.8 + (math.sin(self.steps * 0.3) + 1) * 2)
        glow_color = (*self.COLOR_PLAYER_GLOW, 50)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, glow_color)
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_particles(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            p.draw(s)
        self.screen.blit(s, (0, 0))

    def _render_ui(self):
        ui_surface = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)

        # Time State
        time_text = self.font_time.render(self.TIME_NAMES[self.time_state], True, self.TIME_COLORS[self.time_state])
        ui_surface.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 8))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        ui_surface.blit(score_text, (10, 12))

        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        ui_surface.blit(steps_text, (150, 12))

        # Room
        room_text = self.font_ui.render(f"ROOM: {self.level + 1}/{self.NUM_ROOMS}", True, self.COLOR_UI_TEXT)
        ui_surface.blit(room_text, (300, 12))

        # Materials
        mat_text = self.font_ui.render(f"{self.dream_materials}", True, self.COLOR_MATERIAL)
        pygame.draw.circle(ui_surface, self.COLOR_MATERIAL, (self.WIDTH - 150, 20), 8)
        ui_surface.blit(mat_text, (self.WIDTH - 135, 12))

        # Keys
        key_text = self.font_ui.render(f"{self.keys}", True, self.COLOR_KEY)
        key_rect = pygame.Rect(self.WIDTH - 75, 15, 10, 15)
        pygame.draw.rect(ui_surface, self.COLOR_KEY, key_rect)
        pygame.draw.circle(ui_surface, self.COLOR_KEY, (key_rect.centerx, key_rect.top - 2), 5)
        ui_surface.blit(key_text, (self.WIDTH - 55, 12))
        
        self.screen.blit(ui_surface, (0, 0))

    # --- EFFECT CREATORS ---
    def _create_time_shift_effect(self):
        color = self.TIME_COLORS[self.time_state]
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append(Particle(
                self.WIDTH / 2, self.HEIGHT / 2, color,
                random.uniform(2, 8), 20,
                math.cos(angle) * speed, math.sin(angle) * speed, gravity=0
            ))

    def _create_collect_effect(self, pos):
        px, py = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(20):
            self.particles.append(Particle(
                px, py, self.COLOR_MATERIAL,
                random.uniform(2, 5), 30,
                random.uniform(-2, 2), random.uniform(-3, -1)
            ))

    def _create_crafting_effect(self, pos):
        px, py = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for i in range(36):
            angle = math.radians(i * 10)
            speed = 3
            self.particles.append(Particle(
                px, py, self.COLOR_KEY, 4, 40,
                math.cos(angle) * speed, math.sin(angle) * speed, gravity=0
            ))

    def _create_unlock_effect(self, pos):
        px, py = (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(50):
            self.particles.append(Particle(
                px, py, self.COLOR_DOOR,
                random.uniform(1, 4), 50,
                random.uniform(-4, 4), random.uniform(-4, 4), gravity=0.05
            ))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Dream Labyrinth")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    clock = pygame.time.Clock()

    movement_action = 0
    space_action = 0
    shift_action = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        
        movement_action = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement_action = 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: movement_action = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement_action = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        if terminated:
            font = pygame.font.SysFont("monospace", 48, bold=True)
            text = font.render("EPISODE END", True, (255, 255, 255))
            text_rect = text.get_rect(center=(GameEnv.WIDTH / 2, GameEnv.HEIGHT / 2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()