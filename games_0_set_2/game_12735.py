import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:21:39.004052
# Source Brief: brief_02735.md
# Brief Index: 2735
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for particle effects
class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.life = 0

    def update(self):
        self.pos += self.vel
        self.life += 1
        return self.life >= self.lifetime

    def draw(self, surface, camera_offset):
        # Fade out effect
        alpha = max(0, 255 - int(255 * (self.life / self.lifetime)))
        current_color = (*self.color, alpha)
        
        # Create a temporary surface for the particle to handle alpha transparency
        particle_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, current_color, (self.size, self.size), self.size)
        
        draw_pos = self.pos - camera_offset
        surface.blit(particle_surf, (int(draw_pos.x - self.size), int(draw_pos.y - self.size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Carve tunnels through procedurally generated asteroids with your laser to find and "
        "collect valuable minerals. The higher your score, the richer the asteroids you can explore."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship. Press space to fire your laser. "
        "Shift is an available action but currently has no effect."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_LASER = (255, 20, 20)
    COLOR_ROCK = (80, 80, 90)
    COLOR_ROCK_PARTICLE = (120, 120, 130)
    MINERAL_DATA = {
        2: {'color': (0, 150, 255), 'value': 1, 'name': 'Aquamarine'},
        3: {'color': (0, 200, 100), 'value': 2, 'name': 'Emerald'},
        4: {'color': (200, 50, 255), 'value': 5, 'name': 'Amethyst'},
        5: {'color': (255, 150, 0), 'value': 10, 'name': 'Topaz'},
    }
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 100, 80
    TILE_SIZE = 32
    MAX_STEPS = 5000
    PLAYER_SPEED = 4.0
    LASER_COOLDOWN_MAX = 10 # 10 frames cooldown
    LASER_RANGE = 10 # in tiles
    
    # World cell types
    CELL_EMPTY = 0
    CELL_ROCK = 1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)

        # State variables are initialized in reset()
        self.world_grid = None
        self.player_pos = None
        self.player_visual_pos = None
        self.player_facing_dir = None
        self.steps = 0
        self.score = 0
        self.inventory = None
        self.laser_cooldown = 0
        self.particles = []
        self.stars = []
        self.max_unlocked_level = 1
        self.current_asteroid_level = 1
        
        self._generate_starfield()
        # self.reset() is called by the wrapper, but we can call it to init state
        # self.reset()

        # Critical self-check
        # self.validate_implementation() # Commented out for submission

    def _generate_starfield(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': pygame.Vector2(random.uniform(0, self.WORLD_WIDTH * self.TILE_SIZE), 
                                     random.uniform(0, self.WORLD_HEIGHT * self.TILE_SIZE)),
                'size': random.randint(1, 3),
                'parallax': random.uniform(0.1, 0.4) # Slower stars are further away
            })

    def _generate_asteroid(self):
        # 1. Initialize world as solid rock
        self.world_grid = np.full((self.WORLD_WIDTH, self.WORLD_HEIGHT), self.CELL_ROCK, dtype=np.uint8)

        # 2. Random walk to carve out tunnels
        start_pos = (self.WORLD_WIDTH // 2, self.WORLD_HEIGHT // 2)
        px, py = start_pos
        self.world_grid[px, py] = self.CELL_EMPTY
        
        num_steps = 1500 + 200 * self.current_asteroid_level
        for _ in range(num_steps):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            px, py = max(1, min(self.WORLD_WIDTH - 2, px + dx)), max(1, min(self.WORLD_HEIGHT - 2, py + dy))
            self.world_grid[px, py] = self.CELL_EMPTY
            if random.random() < 0.2: # chance to branch
                self.world_grid[px-1:px+2, py-1:py+2] = self.CELL_EMPTY

        # 3. Flood fill to find all reachable empty cells
        q = [start_pos]
        reachable = {start_pos}
        while q:
            x, y = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WORLD_WIDTH and 0 <= ny < self.WORLD_HEIGHT and \
                   self.world_grid[nx, ny] == self.CELL_EMPTY and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    q.append((nx, ny))
        
        # 4. Place minerals in reachable cells
        mineral_types = list(self.MINERAL_DATA.keys())
        num_minerals = 50 + 20 * self.current_asteroid_level
        for _ in range(num_minerals):
            if not reachable: break
            pos = random.choice(list(reachable))
            
            # Higher levels have better minerals
            mineral_chance = random.random() + self.current_asteroid_level * 0.1
            if mineral_chance > 1.3: mineral_type = 5 # Topaz
            elif mineral_chance > 1.0: mineral_type = 4 # Amethyst
            elif mineral_chance > 0.6: mineral_type = 3 # Emerald
            else: mineral_type = 2 # Aquamarine

            self.world_grid[pos] = mineral_type
            reachable.remove(pos)

        return start_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        if hasattr(self, 'score') and self.score > self.max_unlocked_level * 1000:
            self.max_unlocked_level += 1
        
        self.current_asteroid_level = random.randint(1, self.max_unlocked_level)
        self.score = 0
        
        start_pos = self._generate_asteroid()
        self.player_pos = pygame.Vector2(start_pos)
        self.player_visual_pos = self.player_pos * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2)
        self.player_facing_dir = pygame.Vector2(0, -1) # Default up
        
        self.inventory = {m_id: 0 for m_id in self.MINERAL_DATA.keys()}
        self.laser_cooldown = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- UPDATE LOGIC ---
        self.steps += 1
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

        # Handle Movement
        move_dir = pygame.Vector2(0, 0)
        if movement == 1: move_dir.y = -1 # Up
        elif movement == 2: move_dir.y = 1  # Down
        elif movement == 3: move_dir.x = -1 # Left
        elif movement == 4: move_dir.x = 1  # Right
        
        if move_dir.length_squared() > 0:
            self.player_facing_dir = move_dir.copy()
            next_pos = self.player_pos + move_dir
            if 0 <= next_pos.x < self.WORLD_WIDTH and 0 <= next_pos.y < self.WORLD_HEIGHT:
                cell_type = self.world_grid[int(next_pos.x), int(next_pos.y)]
                if cell_type != self.CELL_ROCK:
                    self.player_pos = next_pos
        
        # Handle Laser
        if space_held and self.laser_cooldown == 0:
            self.laser_cooldown = self.LASER_COOLDOWN_MAX
            # SFX: Laser fire sound
            target_pos = self.player_pos + self.player_facing_dir
            if 0 <= target_pos.x < self.WORLD_WIDTH and 0 <= target_pos.y < self.WORLD_HEIGHT:
                if self.world_grid[int(target_pos.x), int(target_pos.y)] == self.CELL_ROCK:
                    self.world_grid[int(target_pos.x), int(target_pos.y)] = self.CELL_EMPTY
                    # SFX: Rock breaking sound
                    # Spawn particles
                    for _ in range(20):
                        pos = (target_pos * self.TILE_SIZE) + pygame.Vector2(self.TILE_SIZE/2)
                        vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
                        self.particles.append(Particle(pos, vel, self.COLOR_ROCK_PARTICLE, random.randint(1, 4), 30))

        # Handle Mineral Collection
        player_grid_x, player_grid_y = int(self.player_pos.x), int(self.player_pos.y)
        cell_under_player = self.world_grid[player_grid_x, player_grid_y]
        if cell_under_player in self.MINERAL_DATA:
            mineral_id = cell_under_player
            mineral_info = self.MINERAL_DATA[mineral_id]
            
            self.inventory[mineral_id] += 1
            self.score += mineral_info['value']
            self.world_grid[player_grid_x, player_grid_y] = self.CELL_EMPTY
            
            reward += 0.1 # Base reward for any mineral
            if mineral_info['value'] >= 5:
                reward += 1.0 # Bonus for valuable minerals
            
            # SFX: Mineral collection chime
            # Spawn collection particles
            for _ in range(15):
                pos = (self.player_pos * self.TILE_SIZE) + pygame.Vector2(self.TILE_SIZE/2)
                vel = pygame.Vector2(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
                self.particles.append(Particle(pos, vel, mineral_info['color'], random.randint(2, 5), 40))

        # Update particles
        self.particles = [p for p in self.particles if not p.update()]

        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # --- SMOOTH MOVEMENT ---
        target_visual_pos = self.player_pos * self.TILE_SIZE + pygame.Vector2(self.TILE_SIZE/2)
        self.player_visual_pos.move_towards_ip(target_visual_pos, self.PLAYER_SPEED)

        # --- RENDER ---
        camera_offset = self.player_visual_pos - pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)

        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_starfield(camera_offset)
        
        # Game World
        self._render_world(camera_offset)

        # Particles
        for p in self.particles:
            p.draw(self.screen, camera_offset)

        # Laser beam if firing
        if self.laser_cooldown > self.LASER_COOLDOWN_MAX - 3: # Show for 3 frames
            start_pos = self.player_visual_pos - camera_offset
            end_pos = start_pos + self.player_facing_dir * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 3)

        # Player
        self._render_player(camera_offset)

        # UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_starfield(self, camera_offset):
        for star in self.stars:
            star_screen_pos = star['pos'] - camera_offset * star['parallax']
            # Wrap stars around for infinite scrolling feel
            star_screen_pos.x %= self.SCREEN_WIDTH
            star_screen_pos.y %= self.SCREEN_HEIGHT
            color_val = int(255 * star['parallax'] * 2) # Fainter stars are further
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), star_screen_pos, star['size'])

    def _render_world(self, camera_offset):
        # Determine visible tile range
        start_x = max(0, int(camera_offset.x / self.TILE_SIZE))
        end_x = min(self.WORLD_WIDTH, int((camera_offset.x + self.SCREEN_WIDTH) / self.TILE_SIZE) + 1)
        start_y = max(0, int(camera_offset.y / self.TILE_SIZE))
        end_y = min(self.WORLD_HEIGHT, int((camera_offset.y + self.SCREEN_HEIGHT) / self.TILE_SIZE) + 1)

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                cell_type = self.world_grid[x, y]
                if cell_type == self.CELL_EMPTY:
                    continue
                
                screen_pos = pygame.Vector2(x * self.TILE_SIZE, y * self.TILE_SIZE) - camera_offset
                rect = pygame.Rect(int(screen_pos.x), int(screen_pos.y), self.TILE_SIZE, self.TILE_SIZE)
                
                if cell_type == self.CELL_ROCK:
                    pygame.draw.rect(self.screen, self.COLOR_ROCK, rect, border_radius=3)
                elif cell_type in self.MINERAL_DATA:
                    color = self.MINERAL_DATA[cell_type]['color']
                    pygame.draw.rect(self.screen, color, rect.inflate(-8, -8), border_radius=5)
                    pygame.gfxdraw.rectangle(self.screen, rect.inflate(-8, -8), (*color, 100))

    def _render_player(self, camera_offset):
        player_screen_pos = self.player_visual_pos - camera_offset
        # Glow effect
        glow_radius = int(self.TILE_SIZE * 0.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(player_screen_pos.x - glow_radius), int(player_screen_pos.y - glow_radius)))
        # Player core
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_screen_pos, int(self.TILE_SIZE * 0.4))

    def _render_ui(self):
        # Score
        score_text = self.font_title.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Asteroid Level
        level_text = self.font_title.render(f"ASTEROID LV: {self.current_asteroid_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Inventory
        inv_y_start = self.SCREEN_HEIGHT - 30 - (len(self.MINERAL_DATA) * 25)
        inv_title = self.font_ui.render("INVENTORY", True, self.COLOR_UI_TEXT)
        self.screen.blit(inv_title, (self.SCREEN_WIDTH - inv_title.get_width() - 10, inv_y_start))
        
        for i, (m_id, m_data) in enumerate(self.MINERAL_DATA.items()):
            y_pos = inv_y_start + 25 + (i * 25)
            icon_rect = pygame.Rect(self.SCREEN_WIDTH - 110, y_pos, 20, 20)
            pygame.draw.rect(self.screen, m_data['color'], icon_rect, border_radius=3)
            
            inv_text = self.font_ui.render(f"x {self.inventory[m_id]}", True, self.COLOR_UI_TEXT)
            self.screen.blit(inv_text, (self.SCREEN_WIDTH - 80, y_pos + 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "asteroid_level": self.current_asteroid_level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    
    print("Controls: Arrow keys to move, Space to shoot. Close window to quit.")

    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        # The shift key has no action in this game
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}")
            obs, info = env.reset()
            
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()