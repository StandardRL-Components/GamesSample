import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:37:05.049986
# Source Brief: brief_02921.md
# Brief Index: 2921
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a bio-organic server by shifting gravity. Collect data tiles and deliver them to portals while avoiding hostile security programs and falling debris."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to change the direction of gravity and navigate the environment."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors (Cyberpunk/Bio-organic)
    COLOR_BG = (10, 5, 20)
    COLOR_BG_LINES = (30, 20, 50)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 20, 80)
    COLOR_ENEMY_GLOW = (255, 20, 80, 50)
    COLOR_DEBRIS = (0, 180, 255)
    COLOR_DEBRIS_GLOW = (0, 180, 255, 40)
    COLOR_PORTAL_INACTIVE = (255, 128, 0)
    COLOR_PORTAL_ACTIVE = (170, 0, 255)
    COLOR_EXIT = (255, 255, 255)
    TILE_COLORS = [(255, 0, 255), (0, 255, 255), (255, 255, 0)] # Magenta, Cyan, Yellow
    TEXT_COLOR = (220, 220, 240)

    # Physics
    GRAVITY_ACCEL = 0.8
    FRICTION = 0.98
    MAX_VEL = 15
    PATROL_SPEED = 1.5

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

        self.player_rect = None
        self.player_pos = None
        self.player_vel = None
        self.enemies = []
        self.tiles = []
        self.portals = []
        self.debris = []
        self.particles = []
        self.exit_portal = None
        self.all_portals_activated = False
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This should not be here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.all_portals_activated = False
        self.portals_activated_count = 0

        self.gravity_vec = pygame.math.Vector2(0, 1) # Down
        
        self.particles.clear()
        self.debris.clear()
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Player
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 20)
        self.player_rect.center = self.player_pos

        # Portals and Tiles
        self.portals = [
            {'pos': pygame.math.Vector2(50, 50), 'type': 0, 'req': 2, 'count': 0, 'active': False},
            {'pos': pygame.math.Vector2(self.WIDTH - 50, self.HEIGHT - 50), 'type': 1, 'req': 2, 'count': 0, 'active': False}
        ]
        self.tiles = [
            {'pos': pygame.math.Vector2(100, 150), 'vel': pygame.math.Vector2(0,0), 'type': 0, 'size': 12},
            {'pos': pygame.math.Vector2(200, 100), 'vel': pygame.math.Vector2(0,0), 'type': 0, 'size': 12},
            {'pos': pygame.math.Vector2(self.WIDTH - 150, 250), 'vel': pygame.math.Vector2(0,0), 'type': 1, 'size': 12},
            {'pos': pygame.math.Vector2(self.WIDTH - 250, 300), 'vel': pygame.math.Vector2(0,0), 'type': 1, 'size': 12},
        ]
        
        # Enemies
        base_speed = self.PATROL_SPEED + (self.portals_activated_count // 2) * 0.05
        self.enemies = [
            {'pos': pygame.math.Vector2(100, self.HEIGHT - 40), 'vel': pygame.math.Vector2(0,0), 'size': 18, 'patrol_dir': 1, 'speed': base_speed, 'grounded': False}
        ]
        self.exit_portal = None

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        self._handle_action(action)
        
        self._update_physics()
        self._update_enemies()
        self._update_debris()
        self._update_particles()
        
        reward += self._handle_collisions()
        
        terminated, term_reward = self._check_termination_conditions()
        reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action(self, action):
        movement = action[0]
        if movement == 1: self.gravity_vec.update(0, -1) # Up
        elif movement == 2: self.gravity_vec.update(0, 1)  # Down
        elif movement == 3: self.gravity_vec.update(-1, 0) # Left
        elif movement == 4: self.gravity_vec.update(1, 0)  # Right

    def _update_physics(self):
        # Create a list of entities to update, handling player and others differently
        entities_to_update = [
            (self.player_pos, self.player_vel, self.player_rect.width, True)
        ]
        for tile in self.tiles:
            entities_to_update.append((tile['pos'], tile['vel'], tile['size'], False))
        for enemy in self.enemies:
             entities_to_update.append((enemy['pos'], enemy['vel'], enemy['size'], False, enemy))


        for entity_data in entities_to_update:
            pos, vel, size, is_player = entity_data[0:4]
            enemy_ref = entity_data[4] if len(entity_data) > 4 else None

            # Apply Gravity
            vel += self.gravity_vec * self.GRAVITY_ACCEL
            
            # Limit Velocity
            if vel.length() > self.MAX_VEL:
                vel.scale_to_length(self.MAX_VEL)
            
            # Update Position
            pos += vel
            
            # Wall Collisions
            if pos.x - size/2 < 0:
                pos.x = size/2
                vel.x *= -0.5
            if pos.x + size/2 > self.WIDTH:
                pos.x = self.WIDTH - size/2
                vel.x *= -0.5
            if pos.y - size/2 < 0:
                pos.y = size/2
                vel.y *= -0.5
            if pos.y + size/2 > self.HEIGHT:
                pos.y = self.HEIGHT - size/2
                vel.y *= -0.5
                if enemy_ref:
                    enemy_ref['grounded'] = True
            else:
                 if enemy_ref:
                    enemy_ref['grounded'] = False

        self.player_rect.center = self.player_pos

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['grounded']:
                patrol_vec = self.gravity_vec.rotate(90)
                enemy['pos'] += patrol_vec * enemy['patrol_dir'] * enemy['speed']
                
                # Reverse direction at edges
                if self.gravity_vec.y != 0: # Horizontal patrol
                    if enemy['pos'].x <= enemy['size'] or enemy['pos'].x >= self.WIDTH - enemy['size']:
                        enemy['patrol_dir'] *= -1
                else: # Vertical patrol
                    if enemy['pos'].y <= enemy['size'] or enemy['pos'].y >= self.HEIGHT - enemy['size']:
                        enemy['patrol_dir'] *= -1

    def _update_debris(self):
        if self.np_random.random() < 0.03:
            self.debris.append({
                'pos': pygame.math.Vector2(self.np_random.uniform(20, self.WIDTH - 20), -20),
                'vel': pygame.math.Vector2(0, self.np_random.uniform(2, 5)),
                'size': self.np_random.uniform(10, 25)
            })

        for d in self.debris[:]:
            d['pos'] += d['vel']
            if d['pos'].y > self.HEIGHT + d['size']:
                self.debris.remove(d)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Tiles and Portals
        for tile in self.tiles[:]:
            tile_rect = pygame.Rect(0,0, tile['size'], tile['size'])
            tile_rect.center = tile['pos']
            for portal in self.portals:
                if not portal['active'] and portal['type'] == tile['type']:
                    dist = tile['pos'].distance_to(portal['pos'])
                    if dist < 25: # Absorption radius
                        self.tiles.remove(tile)
                        portal['count'] += 1
                        reward += 0.1
                        self.score += 10
                        self._create_particles(portal['pos'], self.TILE_COLORS[portal['type']])
                        break
        
        # Check for portal activation
        for portal in self.portals:
            if not portal['active'] and portal['count'] >= portal['req']:
                portal['active'] = True
                reward += 1.0
                self.score += 100
                self.portals_activated_count += 1

        # Check for level completion
        if not self.all_portals_activated and all(p['active'] for p in self.portals):
            self.all_portals_activated = True
            self.exit_portal = {'pos': pygame.math.Vector2(self.WIDTH/2, 40), 'size': 30}
            
        return reward

    def _check_termination_conditions(self):
        # Player vs Enemy
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(0,0, enemy['size'], enemy['size'])
            enemy_rect.center = enemy['pos']
            if self.player_rect.colliderect(enemy_rect):
                return True, -10.0

        # Player vs Debris
        for d in self.debris:
            debris_rect = pygame.Rect(0,0, d['size'], d['size'])
            debris_rect.center = d['pos']
            if self.player_rect.colliderect(debris_rect):
                return True, -5.0
        
        # Player vs Exit
        if self.exit_portal:
            exit_rect = pygame.Rect(0,0, self.exit_portal['size'], self.exit_portal['size'])
            exit_rect.center = self.exit_portal['pos']
            if self.player_rect.colliderect(exit_rect):
                return True, 100.0

        return False, 0.0

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
                'life': self.np_random.integers(10, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        if self.exit_portal: self._render_exit()
        self._render_portals()
        self._render_tiles()
        self._render_enemies()
        self._render_debris()
        self._render_player()
        self._render_particles()

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_LINES, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_BG_LINES, (0, i), (self.WIDTH, i), 1)

    def _render_portals(self):
        for portal in self.portals:
            color = self.COLOR_PORTAL_ACTIVE if portal['active'] else self.COLOR_PORTAL_INACTIVE
            pos = (int(portal['pos'].x), int(portal['pos'].y))
            radius = 20
            
            if portal['active']:
                pulse = abs(math.sin(self.steps * 0.1))
                glow_radius = int(radius * (1.5 + pulse * 0.5))
                self._draw_glow(pos, glow_radius, (*color, 30))
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

            if not portal['active']:
                req_text = f"{portal['count']}/{portal['req']}"
                text_surf = self.font_small.render(req_text, True, self.TILE_COLORS[portal['type']])
                text_rect = text_surf.get_rect(center=(pos[0], pos[1] + radius + 10))
                self.screen.blit(text_surf, text_rect)

    def _render_tiles(self):
        for tile in self.tiles:
            pos = (int(tile['pos'].x), int(tile['pos'].y))
            size = int(tile['size'] / 2)
            color = self.TILE_COLORS[tile['type']]
            rect = pygame.Rect(pos[0]-size, pos[1]-size, size*2, size*2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = int(enemy['size'])
            self._draw_glow(pos, size + 5, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_trigon(self.screen, pos[0], pos[1]-size//2, pos[0]-size//2, pos[1]+size//2, pos[0]+size//2, pos[1]+size//2, self.COLOR_ENEMY)

    def _render_debris(self):
        for d in self.debris:
            pos = (int(d['pos'].x), int(d['pos'].y))
            size = int(d['size'])
            self._draw_glow(pos, size, self.COLOR_DEBRIS_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size//2, self.COLOR_DEBRIS)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))
        size = self.player_rect.width
        self._draw_glow(pos, size + 5, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=4)
    
    def _render_exit(self):
        if self.exit_portal:
            pos = (int(self.exit_portal['pos'].x), int(self.exit_portal['pos'].y))
            size = self.exit_portal['size']
            pulse = abs(math.sin(self.steps * 0.2))
            
            self._draw_glow(pos, int(size * (1.5 + pulse * 0.5)), (*self.COLOR_EXIT, 40))
            
            rect = pygame.Rect(0,0, size, size)
            rect.center = pos
            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect, 2, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-4, -4), border_radius=5)


    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            try:
                color = (*p['color'], alpha)
                pos = (int(p['pos'].x), int(p['pos'].y))
                size = int(p['life'] / 5)
                if size > 0:
                    pygame.draw.circle(self.screen, color, pos, size)
            except (ValueError, TypeError): # Handle potential color format issues
                pass

    def _render_ui(self):
        score_surf = self.font_small.render(f"SCORE: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_surf, (10, 10))
        
        steps_surf = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.TEXT_COLOR)
        self.screen.blit(steps_surf, (10, 30))

        center = (self.WIDTH - 40, 40)
        arrow_len = 15
        end_pos = (center[0] + self.gravity_vec.x * arrow_len, center[1] + self.gravity_vec.y * arrow_len)
        pygame.draw.line(self.screen, self.TEXT_COLOR, center, end_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 3, self.TEXT_COLOR)
        
        if self.game_over:
            # Check for win condition based on reaching exit portal
            win = False
            if self.exit_portal:
                exit_rect = pygame.Rect(0,0, self.exit_portal['size'], self.exit_portal['size'])
                exit_rect.center = self.exit_portal['pos']
                if self.player_rect.colliderect(exit_rect):
                    win = True
            
            if win:
                 end_text = "SYSTEM ESCAPED"
            else:
                 end_text = "CONNECTION TERMINATED"
            text_surf = self.font_large.render(end_text, True, self.COLOR_PLAYER if win else self.COLOR_ENEMY)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_glow(self, pos, radius, color):
        if radius <= 0: return
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        self.screen.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bio-Organic Server Farm")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    continue
                elif event.key == pygame.K_q:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # The observation is the frame, so we draw it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()