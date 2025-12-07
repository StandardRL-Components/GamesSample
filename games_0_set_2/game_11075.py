import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:34:48.117979
# Source Brief: brief_01075.md
# Brief Index: 1075
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Fighters (Player and Opponent)
class Fighter:
    def __init__(self, x, y, color, radius=15):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.color = color
        self.radius = radius
        self.health = 100.0
        self.energy = 100.0
        self.max_health = 100.0
        self.max_energy = 100.0
        self.speed = 2.5
        self.damping = 0.85
        self.attack_cooldown = 0
        self.damage_flash = 0

    def update(self, screen_width, screen_height):
        # Apply velocity and damping
        self.pos += self.vel
        self.vel *= self.damping

        # Boundary checks
        self.pos.x = np.clip(self.pos.x, self.radius, screen_width - self.radius)
        self.pos.y = np.clip(self.pos.y, self.radius, screen_height - self.radius)

        # Cooldowns
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.damage_flash > 0:
            self.damage_flash -= 1
        
        # Energy regeneration
        self.energy = min(self.max_energy, self.energy + 0.1)

    def move(self, direction):
        if direction == 1:  # Up
            self.vel.y -= self.speed
        elif direction == 2:  # Down
            self.vel.y += self.speed
        elif direction == 3:  # Left
            self.vel.x -= self.speed
        elif direction == 4:  # Right
            self.vel.x += self.speed

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)
        self.damage_flash = 10 # Flash for 10 frames

# Helper class for Particles
class Particle:
    def __init__(self, x, y, color, life, size_range=(2, 5), vel_range=(-2, 2)):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(random.uniform(*vel_range), random.uniform(*vel_range))
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(*size_range)

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A futuristic arena fighter where you battle an opponent on a neon grid. "
        "Match tiles to create teleportation portals and use energy to launch close-range attacks."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press 'space' to select a tile, "
        "then move to a matching tile and press 'space' again to create a portal. "
        "Press 'shift' to attack when near the opponent."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    TILE_W, TILE_H = WIDTH // GRID_COLS, HEIGHT // GRID_ROWS
    MAX_STEPS = 2000
    
    # Colors (Neon Sci-Fi)
    COLOR_BG = (10, 5, 20)
    COLOR_GRID = (30, 20, 50)
    COLOR_PLAYER = (255, 0, 100)
    COLOR_OPPONENT = (0, 150, 255)
    COLOR_PORTAL = (180, 0, 255)
    COLOR_ENERGY = (255, 255, 0)
    COLOR_HEALTH = (0, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    TILE_COLORS = [
        (255, 80, 80),   # Red-ish
        (80, 255, 80),   # Green-ish
        (80, 80, 255),   # Blue-ish
        (255, 255, 80),  # Yellow-ish
        (80, 255, 255),  # Cyan-ish
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables
        self.player = None
        self.opponent = None
        self.tile_grid = []
        self.portals = []
        self.particles = []
        self.selected_tile_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0

        # AI state
        self.ai_attack_speed_factor = 1.0
        
        # self.reset() is called by the wrapper
        # self.validate_implementation() # Can be removed after verification

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = Fighter(100, self.HEIGHT // 2, self.COLOR_PLAYER)
        self.opponent = Fighter(self.WIDTH - 100, self.HEIGHT // 2, self.COLOR_OPPONENT)
        
        self._generate_tile_grid()
        self.portals = []
        self.particles = []
        self.selected_tile_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0
        self.ai_attack_speed_factor = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.player.move(movement)
        
        # Reward for moving closer to opponent
        prev_dist = self.player.pos.distance_to(self.opponent.pos)
        
        if space_press:
            reward += self._handle_portal_creation(self.player)
        if shift_press:
            reward += self._handle_attack(self.player, self.opponent)
            
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Opponent AI ---
        self._update_opponent_ai()
        
        # --- Update Game State ---
        self.player.update(self.WIDTH, self.HEIGHT)
        self.opponent.update(self.WIDTH, self.HEIGHT)
        
        # After move, calculate distance reward
        new_dist = self.player.pos.distance_to(self.opponent.pos)
        if new_dist < prev_dist:
            reward += 0.01

        self._update_portals()
        self._update_particles()

        if self.screen_shake > 0:
            self.screen_shake -= 1

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.ai_attack_speed_factor = max(0.5, self.ai_attack_speed_factor - 0.05)

        # --- Termination and Rewards ---
        terminated = self.player.health <= 0 or self.opponent.health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # Truncated is handled by the wrapper
        if terminated and not self.game_over:
            self.game_over = True
            if self.player.health <= 0 or (self.steps >= self.MAX_STEPS and self.player.health < self.opponent.health):
                reward -= 100 # Lose
            elif self.opponent.health <= 0 or (self.steps >= self.MAX_STEPS and self.player.health > self.opponent.health):
                reward += 100 # Win
        
        # Add damage rewards
        if self.player.damage_flash == 9: # Just got hit
            reward -= 1.0
        if self.opponent.damage_flash == 9: # Just hit opponent
            reward += 1.0

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_tile_at(self, pos):
        grid_x = int(pos.x // self.TILE_W)
        grid_y = int(pos.y // self.TILE_H)
        grid_x = np.clip(grid_x, 0, self.GRID_COLS - 1)
        grid_y = np.clip(grid_y, 0, self.GRID_ROWS - 1)
        return (grid_x, grid_y), self.tile_grid[grid_y][grid_x]

    def _handle_portal_creation(self, entity):
        reward = 0
        tile_pos, tile_color_idx = self._get_tile_at(entity.pos)
        
        if self.selected_tile_pos is None:
            self.selected_tile_pos = tile_pos
            # sfx: select_tile.wav
        else:
            _, selected_tile_color_idx = self._get_tile_at(pygame.math.Vector2((self.selected_tile_pos[0] + 0.5) * self.TILE_W, (self.selected_tile_pos[1] + 0.5) * self.TILE_H))
            if tile_pos != self.selected_tile_pos and tile_color_idx == selected_tile_color_idx:
                # Successful match
                pos1_center = ((self.selected_tile_pos[0] + 0.5) * self.TILE_W, (self.selected_tile_pos[1] + 0.5) * self.TILE_H)
                pos2_center = ((tile_pos[0] + 0.5) * self.TILE_W, (tile_pos[1] + 0.5) * self.TILE_H)
                self.portals.append((pygame.math.Vector2(pos1_center), pygame.math.Vector2(pos2_center)))
                self._create_burst(pos1_center, self.COLOR_PORTAL, 20)
                self._create_burst(pos2_center, self.COLOR_PORTAL, 20)
                reward += 5.0 # Portal creation reward
                # sfx: portal_open.wav
            else:
                # Failed match
                reward -= 0.1
                # sfx: match_fail.wav
            self.selected_tile_pos = None
        return reward

    def _handle_attack(self, attacker, target):
        ATTACK_RANGE = 40
        ATTACK_COST = 25
        ATTACK_DMG = 10
        
        if attacker.energy >= ATTACK_COST and attacker.attack_cooldown == 0:
            if attacker.pos.distance_to(target.pos) < ATTACK_RANGE:
                attacker.energy -= ATTACK_COST
                attacker.attack_cooldown = 30 # 1 second cooldown at 30fps
                target.take_damage(ATTACK_DMG)
                self.screen_shake = 5
                # Create visual effect
                mid_point = attacker.pos.lerp(target.pos, 0.5)
                self._create_burst(mid_point, attacker.color, 30, vel_range=(-4, 4))
                # sfx: hit_success.wav
                return 0 # Reward is handled in step based on damage flash
            else:
                # sfx: attack_whiff.wav
                pass # Miss
        return 0
    
    def _update_opponent_ai(self):
        # AI difficulty scaling
        reaction_time = int(max(0, 10 * (1 - self.steps / 200)))
        if self.steps % (15 + reaction_time) != 0:
            return

        # Simple state machine
        dist_to_player = self.opponent.pos.distance_to(self.player.pos)
        
        # Priority 1: Attack if in range and has energy
        if dist_to_player < 40 and self.opponent.energy > 25 and self.opponent.attack_cooldown == 0:
            self._handle_attack(self.opponent, self.player)
            self.opponent.attack_cooldown = int(45 * self.ai_attack_speed_factor)
        # Priority 2: Move towards player
        else:
            direction_vec = (self.player.pos - self.opponent.pos).normalize()
            if abs(direction_vec.x) > abs(direction_vec.y):
                self.opponent.move(4 if direction_vec.x > 0 else 3)
            else:
                self.opponent.move(2 if direction_vec.y > 0 else 1)

    def _update_portals(self):
        fighters = [self.player, self.opponent]
        for p1, p2 in self.portals:
            for f in fighters:
                if f.pos.distance_to(p1) < f.radius:
                    f.pos = p2.copy()
                    f.vel *= 1.5 # Momentum boost
                    self._create_burst(p2, self.COLOR_PORTAL, 15, vel_range=(-3, 3))
                    # sfx: portal_whoosh.wav
                elif f.pos.distance_to(p2) < f.radius:
                    f.pos = p1.copy()
                    f.vel *= 1.5
                    self._create_burst(p1, self.COLOR_PORTAL, 15, vel_range=(-3, 3))
                    # sfx: portal_whoosh.wav
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _generate_tile_grid(self):
        self.tile_grid = [[random.randint(0, len(self.TILE_COLORS) - 1) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

    def _get_observation(self):
        offset = (0, 0)
        if self.screen_shake > 0:
            offset = (random.randint(-4, 4), random.randint(-4, 4))
        
        # Create a temporary surface for rendering with shake
        render_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        
        # Render all game elements
        self._render_background(render_surface)
        self._render_tiles(render_surface)
        self._render_portals(render_surface)
        self._render_particles(render_surface)
        self._render_fighters(render_surface)
        
        # Blit the shaken surface to the main screen
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(render_surface, offset)

        # Render UI overlay (not affected by shake)
        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- RENDER METHODS ---
    def _draw_glow_circle(self, surface, color, center, radius, blur_factor=3):
        # This is an expensive operation, use sparingly
        target_rect = pygame.Rect(center[0] - radius - blur_factor*2, center[1] - radius - blur_factor*2, (radius + blur_factor*2)*2, (radius + blur_factor*2)*2)
        s = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        
        for i in range(blur_factor, 0, -1):
            alpha = int(150 / (i * 1.5))
            pygame.gfxdraw.filled_circle(s, int(s.get_width()/2), int(s.get_height()/2), int(radius + i * 2), (*color, alpha))
        
        surface.blit(s, target_rect.topleft)


    def _render_background(self, surface):
        surface.fill(self.COLOR_BG)
        for r in range(self.GRID_ROWS):
            pygame.draw.line(surface, self.COLOR_GRID, (0, r * self.TILE_H), (self.WIDTH, r * self.TILE_H))
        for c in range(self.GRID_COLS):
            pygame.draw.line(surface, self.COLOR_GRID, (c * self.TILE_W, 0), (c * self.TILE_W, self.HEIGHT))

    def _render_tiles(self, surface):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.tile_grid[r][c]
                color = self.TILE_COLORS[color_idx]
                rect = pygame.Rect(c * self.TILE_W, r * self.TILE_H, self.TILE_W, self.TILE_H)
                
                # Highlight selected tile
                if self.selected_tile_pos and (c, r) == self.selected_tile_pos:
                    highlight_color = (255, 255, 255)
                    pygame.draw.rect(surface, highlight_color, rect.inflate(2, 2), 2)
                
                # Draw tile with a subtle inner glow
                inner_rect = rect.inflate(-8, -8)
                pygame.gfxdraw.box(surface, inner_rect, (*color, 50))
                pygame.gfxdraw.rectangle(surface, inner_rect, (*color, 150))

    def _render_portals(self, surface):
        for p1, p2 in self.portals:
            self._draw_glow_circle(surface, self.COLOR_PORTAL, p1, 10)
            self._draw_glow_circle(surface, self.COLOR_PORTAL, p2, 10)
            # Draw connecting line with alpha
            pygame.draw.line(surface, (*self.COLOR_PORTAL, 100), p1, p2, 2)
    
    def _render_particles(self, surface):
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(p.pos.x), int(p.pos.y), int(p.size), color)

    def _render_fighters(self, surface):
        for f in [self.opponent, self.player]: # Draw opponent first
            # Damage flash
            render_color = f.color
            if f.damage_flash > 0:
                # Alternate between white and original color for a flicker effect
                render_color = (255, 255, 255) if f.damage_flash % 4 < 2 else f.color

            # Draw fighter with glow
            self._draw_glow_circle(surface, render_color, f.pos, f.radius)
            # Draw core
            pygame.gfxdraw.filled_circle(surface, int(f.pos.x), int(f.pos.y), f.radius - 5, render_color)
            pygame.gfxdraw.filled_circle(surface, int(f.pos.x), int(f.pos.y), f.radius - 8, (255, 255, 255))

    def _render_ui(self, surface):
        # Player 1 UI (Left)
        self._draw_bar(surface, 20, 20, 200, 15, self.player.health / self.player.max_health, self.COLOR_HEALTH, self.COLOR_GRID)
        self._draw_bar(surface, 20, 40, 150, 10, self.player.energy / self.player.max_energy, self.COLOR_ENERGY, self.COLOR_GRID)
        
        # Player 2 UI (Right)
        self._draw_bar(surface, self.WIDTH - 220, 20, 200, 15, self.opponent.health / self.opponent.max_health, self.COLOR_HEALTH, self.COLOR_GRID)
        self._draw_bar(surface, self.WIDTH - 170, 40, 150, 10, self.opponent.energy / self.opponent.max_energy, self.COLOR_ENERGY, self.COLOR_GRID)

        # Score and Steps
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        surface.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        surface.blit(steps_text, (self.WIDTH // 2 - steps_text.get_width() // 2, 40))

    def _draw_bar(self, surface, x, y, w, h, percent, color, bg_color):
        percent = np.clip(percent, 0, 1)
        pygame.draw.rect(surface, bg_color, (x, y, w, h))
        pygame.draw.rect(surface, color, (x, y, w * percent, h))
        pygame.draw.rect(surface, (255, 255, 255), (x, y, w, h), 1)

    def _create_burst(self, pos, color, count, life=20, size_range=(1,4), vel_range=(-2,2)):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, life, size_range, vel_range))
            
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

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv()
    obs, info = env.reset()
    
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Grid Fighter")
    
    running = True
    while running:
        # Human controls
        action = [0, 0, 0] # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            env.reset()

        env.clock.tick(30) # Limit to 30 FPS

    pygame.quit()