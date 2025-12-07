import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set up Pygame to run headlessly
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to navigate your ship. Evade the red aliens and collect yellow fuel cells."
    )

    game_description = (
        "Escape a hostile alien system! Pilot your ship to collect 100 fuel units while evading an ever-growing swarm of patrol drones. Reach the green escape zone with a full tank to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500
    WIN_FUEL = 100

    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_ALIEN_GLOW = (255, 100, 100, 60)
    COLOR_FUEL = (255, 220, 0)
    COLOR_FUEL_GLOW = (255, 220, 0, 70)
    COLOR_ESCAPE_ZONE = (0, 255, 100)
    COLOR_ESCAPE_ZONE_INACTIVE = (0, 100, 50)
    COLOR_HEALTH_BAR = (0, 200, 80)
    COLOR_HEALTH_BAR_BG = (80, 0, 20)
    COLOR_TEXT = (220, 220, 220)

    PLAYER_SPEED = 5
    PLAYER_SIZE = 10
    PLAYER_MAX_HEALTH = 100
    
    INITIAL_ALIENS = 2
    MAX_ALIENS = 10
    ALIEN_ADD_INTERVAL = 200
    ALIEN_SPEED_INCREASE_INTERVAL = 100
    ALIEN_SPEED_INCREASE_AMOUNT = 0.05
    
    INITIAL_FUEL_CELLS = 40

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.player = {}
        self.aliens = []
        self.fuel_cells = []
        self.particles = []
        self.escape_zone = None
        self.steps = 0
        self.fuel = 0
        self.game_over = False
        self.win = False
        self.alien_base_speed = 1.0
        
        self.np_random = None # Will be seeded in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.fuel = 0
        self.game_over = False
        self.win = False
        self.alien_base_speed = 1.0

        self.player = {
            'x': self.WIDTH / 2,
            'y': self.HEIGHT / 2,
            'health': self.PLAYER_MAX_HEALTH,
            'angle': -math.pi / 2,
            'last_move': 0
        }
        
        self.escape_zone = pygame.Rect(self.WIDTH - 80, 10, 70, self.HEIGHT - 20)
        
        self.aliens = []
        for _ in range(self.INITIAL_ALIENS):
            self._add_alien()
            
        self.fuel_cells = []
        while len(self.fuel_cells) < self.INITIAL_FUEL_CELLS:
            cell = pygame.Rect(
                self.np_random.integers(20, self.WIDTH - 20),
                self.np_random.integers(20, self.HEIGHT - 20),
                10, 10
            )
            # Avoid spawning on player or in escape zone
            player_spawn_rect = pygame.Rect(self.player['x']-20, self.player['y']-20, 40, 40)
            if not cell.colliderect(player_spawn_rect) and \
               not cell.colliderect(self.escape_zone):
                self.fuel_cells.append(cell)
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Time penalty

        if not self.game_over:
            movement = action[0]
            self._move_player(movement)
            self._update_aliens()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._update_difficulty()
            self._update_particles()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.game_over = True
        
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_player(self, movement):
        if movement != 0:
            self.player['last_move'] = movement
        
        if movement == 1:  # Up
            self.player['y'] -= self.PLAYER_SPEED
            self.player['angle'] = -math.pi / 2
        elif movement == 2:  # Down
            self.player['y'] += self.PLAYER_SPEED
            self.player['angle'] = math.pi / 2
        elif movement == 3:  # Left
            self.player['x'] -= self.PLAYER_SPEED
            self.player['angle'] = math.pi
        elif movement == 4:  # Right
            self.player['x'] += self.PLAYER_SPEED
            self.player['angle'] = 0

        self.player['x'] = np.clip(self.player['x'], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player['y'] = np.clip(self.player['y'], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_aliens(self):
        for alien in self.aliens:
            if alien['axis'] == 'x':
                alien['x'] += alien['speed'] * alien['dir']
                if alien['x'] > alien['patrol_end'] or alien['x'] < alien['patrol_start']:
                    alien['dir'] *= -1
                    alien['x'] = np.clip(alien['x'], alien['patrol_start'], alien['patrol_end'])
            else:
                alien['y'] += alien['speed'] * alien['dir']
                if alien['y'] > alien['patrol_end'] or alien['y'] < alien['patrol_start']:
                    alien['dir'] *= -1
                    alien['y'] = np.clip(alien['y'], alien['patrol_start'], alien['patrol_end'])

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player['x'] - self.PLAYER_SIZE/2, self.player['y'] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player vs Fuel
        collected_indices = []
        for i, cell in enumerate(self.fuel_cells):
            if player_rect.colliderect(cell):
                collected_indices.append(i)
                
                for _ in range(10):
                    self._add_particle(cell.centerx, cell.centery, self.COLOR_FUEL)
        
        if collected_indices:
            self.fuel_cells = [cell for i, cell in enumerate(self.fuel_cells) if i not in collected_indices]
            
            fuel_before = self.fuel
            self.fuel += len(collected_indices)
            reward += 0.1 * len(collected_indices)
            
            if fuel_before < 50 <= self.fuel:
                reward += 1.0
            if fuel_before < self.WIN_FUEL <= self.fuel:
                reward += 5.0
            
            self.fuel = min(self.fuel, self.WIN_FUEL + 10) # Cap fuel slightly above win condition

        # Player vs Aliens
        for alien in self.aliens:
            dist = math.hypot(self.player['x'] - alien['x'], self.player['y'] - alien['y'])
            if dist < self.PLAYER_SIZE / 2 + alien['radius']:
                self.player['health'] -= 1
                for _ in range(15):
                    self._add_particle(self.player['x'], self.player['y'], self.COLOR_ALIEN)
        
        self.player['health'] = max(0, self.player['health'])
        return reward

    def _update_difficulty(self):
        if self.steps > 0:
            if self.steps % self.ALIEN_SPEED_INCREASE_INTERVAL == 0:
                self.alien_base_speed = min(3.0, self.alien_base_speed + self.ALIEN_SPEED_INCREASE_AMOUNT)

            if self.steps % self.ALIEN_ADD_INTERVAL == 0 and len(self.aliens) < self.MAX_ALIENS:
                self._add_alien()

    def _add_alien(self):
        side = self.np_random.integers(0, 4)
        if side == 0: x, y = 0, self.np_random.integers(0, self.HEIGHT)
        elif side == 1: x, y = self.WIDTH, self.np_random.integers(0, self.HEIGHT)
        elif side == 2: x, y = self.np_random.integers(0, self.WIDTH), 0
        else: x, y = self.np_random.integers(0, self.WIDTH), self.HEIGHT
        
        radius = self.np_random.integers(8, 16)
        speed_multiplier = 1 + (12 - radius) / 8 # Smaller aliens are faster
        axis = 'x' if self.np_random.random() > 0.5 else 'y'
        patrol_range = self.np_random.integers(100, 300)
        
        self.aliens.append({
            'x': x, 'y': y, 'radius': radius,
            'speed': self.alien_base_speed * speed_multiplier,
            'axis': axis, 'dir': 1,
            'patrol_start': x - patrol_range/2 if axis == 'x' else y - patrol_range/2,
            'patrol_end': x + patrol_range/2 if axis == 'x' else y + patrol_range/2,
        })
        
    def _check_termination(self):
        if self.player['health'] <= 0:
            self.win = False
            return True
        
        player_rect = pygame.Rect(self.player['x'], self.player['y'], 1, 1)
        if self.fuel >= self.WIN_FUEL and self.escape_zone.contains(player_rect):
            self.win = True
            return True
            
        return False

    def _add_particle(self, x, y, color):
        self.particles.append({
            'x': x, 'y': y,
            'vx': self.np_random.random() * 4 - 2,
            'vy': self.np_random.random() * 4 - 2,
            'lifespan': self.np_random.integers(10, 20),
            'color': color
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Escape Zone
        is_active = self.fuel >= self.WIN_FUEL
        zone_color = self.COLOR_ESCAPE_ZONE if is_active else self.COLOR_ESCAPE_ZONE_INACTIVE
        pygame.draw.rect(self.screen, zone_color, self.escape_zone, 2, border_radius=5)
        
        # Fuel Cells
        for cell in self.fuel_cells:
            glow_rect = cell.inflate(6, 6)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_FUEL_GLOW, s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_FUEL, cell, border_radius=3)
            
        # Aliens
        for alien in self.aliens:
            pos = (int(alien['x']), int(alien['y']))
            radius = int(alien['radius'])
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius * 1.5), self.COLOR_ALIEN_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ALIEN)
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20))
            color = (*p['color'], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (p['x'], p['y']))

        # Player
        px, py = int(self.player['x']), int(self.player['y'])
        angle = self.player['angle']
        points = [
            (px + self.PLAYER_SIZE * math.cos(angle), py + self.PLAYER_SIZE * math.sin(angle)),
            (px + self.PLAYER_SIZE * math.cos(angle + 2.2), py + self.PLAYER_SIZE * math.sin(angle + 2.2)),
            (px + self.PLAYER_SIZE * math.cos(angle - 2.2), py + self.PLAYER_SIZE * math.sin(angle - 2.2)),
        ]
        # Glow
        pygame.gfxdraw.filled_trigon(self.screen, 
            int(points[0][0]), int(points[0][1]), 
            int(points[1][0]), int(points[1][1]), 
            int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
    def _render_ui(self):
        # Health Bar
        health_ratio = self.player['health'] / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))
        
        # Fuel Counter
        fuel_text = self.font_small.render(f"FUEL: {self.fuel}/{self.WIN_FUEL}", True, self.COLOR_FUEL)
        self.screen.blit(fuel_text, (self.WIDTH - fuel_text.get_width() - 10, 10))
        
        # Game Over / Win Text
        if self.game_over:
            text_str = "MISSION COMPLETE" if self.win else "GAME OVER"
            color = self.COLOR_ESCAPE_ZONE if self.win else self.COLOR_ALIEN
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(text_str, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "fuel": self.fuel,
            "health": self.player['health'],
            "steps": self.steps,
            "aliens": len(self.aliens)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Requires pygame to be installed with display support
    os.environ['SDL_VIDEODRIVER'] = 'x11' # or 'windows', 'mac', etc.
    import sys

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Alien Escape")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()
    pygame.quit()
    sys.exit()