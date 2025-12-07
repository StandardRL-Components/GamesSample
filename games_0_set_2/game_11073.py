import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:27:19.436141
# Source Brief: brief_01073.md
# Brief Index: 1073
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a tile-matching combat game.
    
    The agent controls a cursor on a tile grid. Matching 3 or more tiles of the
    same color charges a corresponding portal. When a portal is fully charged,
    it automatically spawns a fighter. Fighters then move to the combat arena
    and automatically attack enemy fighters.

    The goal is to defeat all enemy fighters. The episode ends if all player
    fighters are defeated, all enemy fighters are defeated, or the time limit
    is reached.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Match tiles on the grid to charge portals and summon fighters. "
        "Your fighters will automatically battle enemies in the arena."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select a tile, "
        "then move to an adjacent tile and press space again to swap."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen and Layout
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 8
    TILE_SIZE = 40
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    GRID_X = 40
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    ARENA_X = GRID_X + GRID_WIDTH + 20
    ARENA_WIDTH = SCREEN_WIDTH - ARENA_X - 20
    
    # Colors
    COLOR_BG = pygame.Color("#1E1E2E")
    COLOR_GRID_BG = pygame.Color("#181825")
    COLOR_GRID_LINE = pygame.Color("#313244")
    COLOR_ARENA_BG = pygame.Color("#111118")
    COLOR_TEXT = pygame.Color("#CDD6F4")
    COLOR_PLAYER = pygame.Color("#89DCEB")
    COLOR_ENEMY = pygame.Color("#F38BA8")
    COLOR_PLAYER_PROJECTILE = pygame.Color("#94E2D5")
    COLOR_ENEMY_PROJECTILE = pygame.Color("#FAB387")
    TILE_COLORS = [
        pygame.Color("#F38BA8"),  # Red
        pygame.Color("#A6E3A1"),  # Green
        pygame.Color("#89B4FA"),  # Blue
        pygame.Color("#F9E2AF"),  # Yellow
        pygame.Color("#CBA6F7"),  # Mauve
    ]

    # Game Parameters
    MAX_STEPS = 1500
    PORTAL_CHARGE_COST = 10
    FIGHTER_SPAWN_Y = SCREEN_HEIGHT // 2
    FIGHTER_SPEED = 1.5
    FIGHTER_ATTACK_RANGE = 120
    FIGHTER_ATTACK_COOLDOWN = 60 # frames
    FIGHTER_BASE_HEALTH = 100
    PROJECTILE_SPEED = 5
    PROJECTILE_DAMAGE = 10
    NUM_INITIAL_ENEMIES = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Non-resetting state
        self.wins = 0
        self.prev_space_held = False

        # self.reset() is called by the wrapper or user
        # self.validate_implementation() is for debugging, not needed in production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = "" # "WIN" or "LOSE"

        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_tile_pos = None
        
        self.portal_power = {i: 0 for i in range(len(self.TILE_COLORS))}
        
        self.player_fighters = []
        self.enemy_fighters = []
        self.projectiles = []
        self.particles = []

        self.match_animation_timer = 0
        self.matched_tiles = set()
        
        self.current_reward = 0.0

        self._generate_grid()
        self._spawn_initial_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.current_reward = 0.0
        self.steps += 1
        
        if not self.game_over:
            self._handle_input(action)
            self._update_grid_logic()
            self._update_combat_logic()
            self._check_spawns()
            self.score += self.current_reward

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            self.game_outcome = "TIME UP"
        
        return (
            self._get_observation(),
            self.current_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_raw, shift_raw = action
        space_held = space_raw == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- Movement ---
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

        # --- Tile Selection/Swap ---
        if space_pressed and self.match_animation_timer == 0:
            if self.selected_tile_pos is None:
                self.selected_tile_pos = list(self.cursor_pos)
                # Sound: select_tile.wav
            else:
                if self._are_adjacent(self.selected_tile_pos, self.cursor_pos):
                    self._swap_tiles(self.selected_tile_pos, self.cursor_pos)
                    matches = self._find_all_matches()
                    if not matches:
                        # Invalid swap, swap back
                        self._swap_tiles(self.selected_tile_pos, self.cursor_pos)
                        # Sound: invalid_swap.wav
                    else:
                        self._process_matches(matches)
                        # Sound: match_success.wav
                self.selected_tile_pos = None

    def _update_grid_logic(self):
        if self.match_animation_timer > 0:
            self.match_animation_timer -= 1
            if self.match_animation_timer == 0:
                self._clear_and_drop_tiles()
                matches = self._find_all_matches()
                if matches:
                    self._process_matches(matches) # Cascade
                    # Sound: cascade.wav
        
    def _update_combat_logic(self):
        # Update fighters
        for f in self.player_fighters + self.enemy_fighters:
            f.update(self.player_fighters, self.enemy_fighters, self.projectiles)

        # Update projectiles
        new_projectiles = []
        for p in self.projectiles:
            p.update()
            if not p.is_out_of_bounds(self.SCREEN_WIDTH, self.SCREEN_HEIGHT):
                collided = p.check_collision(self.player_fighters + self.enemy_fighters)
                if collided:
                    if not p.is_player_owned: # Player projectile hit enemy
                        self.current_reward += 5.0  # Reward for damaging enemy
                    self._create_impact_particles(p.pos)
                    # Sound: impact.wav
                else:
                    new_projectiles.append(p)
        self.projectiles = new_projectiles
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Remove KO'd fighters
        if any(f.health <= 0 for f in self.enemy_fighters):
            self.current_reward += 10.0 * sum(1 for f in self.enemy_fighters if f.health <= 0) # Reward for KO
            self.enemy_fighters = [f for f in self.enemy_fighters if f.health > 0]
        
        if any(f.health <= 0 for f in self.player_fighters):
            self.player_fighters = [f for f in self.player_fighters if f.health > 0]

    def _check_spawns(self):
        for color_idx, power in self.portal_power.items():
            if power >= self.PORTAL_CHARGE_COST:
                self.portal_power[color_idx] -= self.PORTAL_CHARGE_COST
                self._spawn_player_fighter(color_idx)
                self.current_reward += 0.5
                # Sound: spawn_fighter.wav

    def _check_termination(self):
        if self.game_over:
            return True

        if len(self.enemy_fighters) == 0 and self.steps > 0: # Win condition
            self.game_over = True
            self.game_outcome = "WIN"
            self.current_reward += 100.0
            self.wins += 1
            return True

        # Loss condition: no player fighters and no way to make more
        can_spawn = any(p >= self.PORTAL_CHARGE_COST for p in self.portal_power.values())
        can_match = len(self._find_possible_moves()) > 0
        if len(self.player_fighters) == 0 and not can_spawn and not can_match:
            self.game_over = True
            self.game_outcome = "LOSE"
            self.current_reward -= 100.0
            return True
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_fighters": len(self.player_fighters),
            "enemy_fighters": len(self.enemy_fighters),
            "wins": self.wins
        }

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_grid_area()
        self._render_arena()
        self._render_portals()
        
        for p in self.particles: p.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        for f in self.player_fighters + self.enemy_fighters: f.draw(self.screen)

    def _render_grid_area(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), border_radius=8)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_val = self.grid[r][c]
                if tile_val == -1: continue

                rect = pygame.Rect(self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                
                # Shrink animation for matched tiles
                if (r, c) in self.matched_tiles:
                    scale = self.match_animation_timer / 15.0
                    rect.inflate_ip(-self.TILE_SIZE * (1-scale), -self.TILE_SIZE * (1-scale))

                pygame.draw.rect(self.screen, self.TILE_COLORS[tile_val], rect.inflate(-4,-4), border_radius=5)

        # Draw selected tile highlight
        if self.selected_tile_pos is not None:
            r, c = self.selected_tile_pos
            rect = (self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, pygame.Color("white"), rect, 2, border_radius=7)

        # Draw cursor
        r, c = self.cursor_pos
        rect = (self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        cursor_color = pygame.Color(255, 255, 255, int(alpha))
        gfx_rect = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(gfx_rect, cursor_color, (0, 0, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=7)
        self.screen.blit(gfx_rect, rect)

    def _render_arena(self):
        arena_rect = (self.ARENA_X, self.GRID_Y, self.ARENA_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_ARENA_BG, arena_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, arena_rect, 2, border_radius=8)

    def _render_portals(self):
        portal_area_x = self.GRID_X - 30
        for i, color in enumerate(self.TILE_COLORS):
            y_pos = self.GRID_Y + i * (self.TILE_SIZE + 10) + 15
            power = self.portal_power[i]
            
            # Background bar
            pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (portal_area_x, y_pos, 20, self.TILE_SIZE), border_radius=5)
            
            # Power fill
            fill_height = min(1.0, power / self.PORTAL_CHARGE_COST) * self.TILE_SIZE
            if fill_height > 0:
                pygame.draw.rect(self.screen, color, (portal_area_x, y_pos + self.TILE_SIZE - fill_height, 20, fill_height), border_radius=5)
            
            # Border
            pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (portal_area_x, y_pos, 20, self.TILE_SIZE), 1, border_radius=5)

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_medium.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        outcome_text = self.font_large.render(self.game_outcome, True, pygame.Color("white"))
        text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(outcome_text, text_rect)
        
        self.screen.blit(overlay, (0, 0))

    # --- Grid Logic Helpers ---
    def _generate_grid(self):
        while True:
            self.grid = [[self.np_random.integers(0, len(self.TILE_COLORS)) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.grid[r][c]
                if color == -1: continue
                # Horizontal
                if c < self.GRID_COLS - 2 and self.grid[r][c+1] == color and self.grid[r][c+2] == color:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_ROWS - 2 and self.grid[r+1][c] == color and self.grid[r+2][c] == color:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Swap right
                if c < self.GRID_COLS - 1:
                    self._swap_tiles((r, c), (r, c+1))
                    if self._find_all_matches(): moves.append(((r, c), (r, c+1)))
                    self._swap_tiles((r, c), (r, c+1)) # Swap back
                # Swap down
                if r < self.GRID_ROWS - 1:
                    self._swap_tiles((r, c), (r+1, c))
                    if self._find_all_matches(): moves.append(((r, c), (r+1, c)))
                    self._swap_tiles((r, c), (r+1, c)) # Swap back
        return moves

    def _process_matches(self, matches):
        self.current_reward += 0.1 * len(matches)
        for r, c in matches:
            color_idx = self.grid[r][c]
            if color_idx != -1:
                self.portal_power[color_idx] += 1
        self.matched_tiles.update(matches)
        self.match_animation_timer = 15 # frames

    def _clear_and_drop_tiles(self):
        for r_m, c_m in self.matched_tiles:
            self._create_match_particles((r_m, c_m))
            self.grid[r_m][c_m] = -1

        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    self._swap_tiles((r, c), (empty_row, c))
                    empty_row -= 1
        
        self._refill_grid()
        self.matched_tiles.clear()

    def _refill_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == -1:
                    self.grid[r][c] = self.np_random.integers(0, len(self.TILE_COLORS))

    def _swap_tiles(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

    def _are_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    # --- Spawning and Entity Helpers ---
    def _spawn_player_fighter(self, color_idx):
        pos = np.array([self.ARENA_X + 20, self.FIGHTER_SPAWN_Y + self.np_random.uniform(-50, 50)])
        fighter = Fighter(pos, self.FIGHTER_BASE_HEALTH, self.TILE_COLORS[color_idx], is_player=True, env=self)
        self.player_fighters.append(fighter)

    def _spawn_initial_enemies(self):
        health = self.FIGHTER_BASE_HEALTH * (1 + 0.05 * self.wins)
        for i in range(self.NUM_INITIAL_ENEMIES):
            pos = np.array([self.SCREEN_WIDTH - 40, self.FIGHTER_SPAWN_Y + self.np_random.uniform(-80, 80) * (i+1)])
            fighter = Fighter(pos, health, self.COLOR_ENEMY, is_player=False, env=self)
            self.enemy_fighters.append(fighter)

    def _create_match_particles(self, grid_pos):
        r, c = grid_pos
        x = self.GRID_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        y = self.GRID_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        color = self.TILE_COLORS[self.grid[r][c]]
        for _ in range(10):
            self.particles.append(Particle(np.array([x,y]), color, self.np_random))

    def _create_impact_particles(self, pos):
        for _ in range(15):
            self.particles.append(Particle(pos.copy(), pygame.Color("white"), self.np_random, max_life=15))

# Helper classes for game entities
class Fighter:
    def __init__(self, pos, health, color, is_player, env):
        self.pos = pos
        self.health = health
        self.max_health = health
        self.color = color
        self.is_player = is_player
        self.env = env
        self.target = None
        self.attack_cooldown = 0

    def update(self, player_fighters, enemy_fighters, projectiles):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        enemies = enemy_fighters if self.is_player else player_fighters
        if not enemies:
            self.target = None
            return

        # Find target if needed
        if self.target is None or self.target.health <= 0:
            self.target = min(enemies, key=lambda e: np.linalg.norm(self.pos - e.pos))

        # Move towards target
        direction = self.target.pos - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > self.env.FIGHTER_ATTACK_RANGE:
            self.pos += (direction / dist) * self.env.FIGHTER_SPEED
        elif self.attack_cooldown == 0:
            self.attack()

    def attack(self):
        self.attack_cooldown = self.env.FIGHTER_ATTACK_COOLDOWN
        proj_color = self.env.COLOR_PLAYER_PROJECTILE if self.is_player else self.env.COLOR_ENEMY_PROJECTILE
        projectile = Projectile(self.pos.copy(), self.target, proj_color, self.is_player, self.env)
        self.env.projectiles.append(projectile)
        # Sound: shoot.wav

    def take_damage(self, amount):
        self.health -= amount

    def draw(self, screen):
        # Body
        if self.is_player:
            points = [
                (self.pos[0], self.pos[1] - 10),
                (self.pos[0] - 8, self.pos[1] + 8),
                (self.pos[0] + 8, self.pos[1] + 8)
            ]
            pygame.gfxdraw.aapolygon(screen, [(int(p[0]), int(p[1])) for p in points], self.env.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(screen, [(int(p[0]), int(p[1])) for p in points], self.env.COLOR_PLAYER)
        else:
            rect = pygame.Rect(self.pos[0] - 8, self.pos[1] - 8, 16, 16)
            pygame.draw.rect(screen, self.env.COLOR_ENEMY, rect, border_radius=3)
        
        # Health bar
        bar_width = 30
        bar_height = 5
        bar_y = self.pos[1] - 25
        health_pct = max(0, self.health / self.max_health)
        
        bg_rect = pygame.Rect(self.pos[0] - bar_width/2, bar_y, bar_width, bar_height)
        pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=2)
        
        fill_color = (100, 200, 100) if health_pct > 0.5 else ((200, 200, 100) if health_pct > 0.2 else (200, 100, 100))
        fill_rect = pygame.Rect(self.pos[0] - bar_width/2, bar_y, bar_width * health_pct, bar_height)
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=2)

class Projectile:
    def __init__(self, pos, target, color, is_player_owned, env):
        self.pos = pos
        self.target = target
        self.color = color
        self.is_player_owned = is_player_owned
        self.env = env
        direction = target.pos - pos
        dist = np.linalg.norm(direction)
        self.vel = (direction / dist) * self.env.PROJECTILE_SPEED if dist > 0 else np.array([0,0])

    def update(self):
        self.pos += self.vel

    def check_collision(self, fighters):
        targets = self.env.enemy_fighters if self.is_player_owned else self.env.player_fighters
        for f in targets:
            if np.linalg.norm(self.pos - f.pos) < 10:
                f.take_damage(self.env.PROJECTILE_DAMAGE)
                return True
        return False

    def is_out_of_bounds(self, w, h):
        return not (0 < self.pos[0] < w and 0 < self.pos[1] < h)

    def draw(self, screen):
        pygame.gfxdraw.aacircle(screen, int(self.pos[0]), int(self.pos[1]), 4, self.color)
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), 4, self.color)

class Particle:
    def __init__(self, pos, color, np_random, max_life=30):
        self.pos = pos
        self.color = color
        self.np_random = np_random
        self.life = max_life
        self.max_life = max_life
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        self.vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.life -= 1
        return self.life > 0

    def draw(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        size = int(5 * (self.life / self.max_life))
        if size > 0:
            temp_color = self.color.copy()
            temp_color.a = alpha
            
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, temp_color, (size, size), size)
            screen.blit(particle_surf, (int(self.pos[0] - size), int(self.pos[1] - size)))


if __name__ == '__main__':
    # Example of how to run the environment
    # This part requires a display. Set SDL_VIDEODRIVER to something other than "dummy".
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Example ---
    # Use arrow keys for movement, space to select/swap, left-shift to do nothing
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play rendering
    pygame.display.set_caption("Tile Combat Arena")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    
    while not done:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {info['score']}, Outcome: {env.game_outcome}")
    pygame.quit()