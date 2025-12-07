import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:56:16.589481
# Source Brief: brief_02001.md
# Brief Index: 2001
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced, side-scrolling shooter set in a neon cyberspace. "
        "Shoot obstacles to gain energy and switch modes to phase through threats."
    )
    user_guide = (
        "Controls: Press Space to shoot. Press Shift to switch between Attack and Dodge modes."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen and World
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    WORLD_SPEED = 2.0

    # Player
    PLAYER_X_POS = 120
    PLAYER_RADIUS = 15
    PLAYER_HEALTH_MAX = 100.0
    PLAYER_ENERGY_MAX = 100.0
    PLAYER_STATE_ATTACK = "ATTACK"
    PLAYER_STATE_DODGE = "DODGE"
    
    # Projectiles
    PROJECTILE_RADIUS = 4
    PROJECTILE_SPEED = 12.0
    PROJECTILE_ENERGY_COST = 8.0
    SHOOT_COOLDOWN_MAX = 8 # frames

    # Obstacles
    OBSTACLE_RADIUS = 20
    OBSTACLE_SPAWN_PROB_INIT = 0.02
    OBSTACLE_BASE_SPEED_INIT = 3.0
    OBSTACLE_TYPE_DMG = "DAMAGE"
    OBSTACLE_TYPE_NRG = "ENERGY"
    OBSTACLE_TYPE_PHASE = "PHASE"

    # Particles
    PARTICLE_RADIUS = 5
    PARTICLE_SPEED = 4.0
    PARTICLE_LIFESPAN = 90 # frames
    PARTICLE_ENERGY_GAIN = 5.0

    # Colors (Vibrant Neon)
    COLOR_BG_START = (10, 0, 20)
    COLOR_BG_END = (30, 0, 50)
    COLOR_GRID = (60, 20, 100, 100)
    COLOR_PLAYER_ATTACK = (0, 255, 255)
    COLOR_PLAYER_DODGE = (50, 255, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_OBSTACLE_DMG = (255, 0, 128)
    COLOR_OBSTACLE_NRG = (0, 150, 255)
    COLOR_OBSTACLE_PHASE = (50, 255, 50)
    COLOR_PARTICLE = (100, 200, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_UI_HEALTH = (50, 200, 50)
    COLOR_UI_HEALTH_LOW = (255, 50, 50)
    COLOR_UI_ENERGY = (0, 150, 255)
    COLOR_UI_FRAME = (150, 150, 200)
    COLOR_UI_BG = (20, 20, 40, 180)

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 32)
        
        # Pre-render background for performance
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self._create_background()
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_energy = 0
        self.player_state = ""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacles = []
        self.projectiles = []
        self.particles = []
        self.grid_lines = []
        self.prev_shift_state = 0
        self.shoot_cooldown = 0
        self.obstacle_base_speed = 0
        self.obstacle_spawn_prob = 0
        self.screen_shake = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.PLAYER_X_POS, self.HEIGHT / 2]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_energy = self.PLAYER_ENERGY_MAX / 2
        self.player_state = self.PLAYER_STATE_ATTACK

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.obstacles = []
        self.projectiles = []
        self.particles = []
        
        self.grid_lines = [
            {'pos': self.np_random.integers(0, self.WIDTH), 'width': self.np_random.integers(1, 4)}
            for _ in range(30)
        ]

        self.prev_shift_state = 0
        self.shoot_cooldown = 0
        
        self.obstacle_base_speed = self.OBSTACLE_BASE_SPEED_INIT
        self.obstacle_spawn_prob = self.OBSTACLE_SPAWN_PROB_INIT
        
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        self._handle_actions(action)
        self._update_world()
        reward += self._handle_collisions()
        self._spawn_obstacles()
        
        self.steps += 1
        self._update_difficulty()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100
            self.game_over = True
            truncated = True # Truncated because it's a time limit
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        _movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # State switching (on rising edge of shift)
        if shift_held and not self.prev_shift_state:
            self.player_state = self.PLAYER_STATE_DODGE if self.player_state == self.PLAYER_STATE_ATTACK else self.PLAYER_STATE_ATTACK
            # SFX: State switch sound
        self.prev_shift_state = shift_held

        # Shooting
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        
        if space_held and self.player_state == self.PLAYER_STATE_ATTACK and self.shoot_cooldown <= 0:
            if self.player_energy >= self.PROJECTILE_ENERGY_COST:
                self.player_energy -= self.PROJECTILE_ENERGY_COST
                self.projectiles.append({
                    'pos': list(self.player_pos),
                    'radius': self.PROJECTILE_RADIUS
                })
                self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX
                # SFX: Player shoot sound
            # No penalty for trying to shoot w/o energy, just inaction.
            
    def _update_world(self):
        # Update projectiles
        for p in self.projectiles[:]:
            p['pos'][0] += self.PROJECTILE_SPEED
            if p['pos'][0] > self.WIDTH + p['radius']:
                self.projectiles.remove(p)

        # Update obstacles
        for o in self.obstacles[:]:
            o['pos'][0] -= o['speed']
            if o['pos'][0] < -o['radius']:
                self.obstacles.remove(o)

        # Update particles
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            dx, dy = self.player_pos[0] - p['pos'][0], self.player_pos[1] - p['pos'][1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                p['pos'][0] += (dx / dist) * self.PARTICLE_SPEED
                p['pos'][1] += (dy / dist) * self.PARTICLE_SPEED

    def _handle_collisions(self):
        reward = 0
        
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)

        # Player vs Obstacles
        for o in self.obstacles[:]:
            obstacle_rect = pygame.Rect(o['pos'][0] - o['radius'], o['pos'][1] - o['radius'], o['radius'] * 2, o['radius'] * 2)
            if player_rect.colliderect(obstacle_rect):
                collided = False
                if o['type'] == self.OBSTACLE_TYPE_DMG:
                    collided = True
                elif o['type'] == self.OBSTACLE_TYPE_NRG:
                    collided = True
                elif o['type'] == self.OBSTACLE_TYPE_PHASE and self.player_state == self.PLAYER_STATE_ATTACK:
                    collided = True

                if collided:
                    self.player_health -= 25.0
                    self.screen_shake = 10
                    reward -= 0.5
                    self.obstacles.remove(o)
                    # SFX: Player hit sound
        
        # Projectiles vs Obstacles
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p['pos'][0] - p['radius'], p['pos'][1] - p['radius'], p['radius'] * 2, p['radius'] * 2)
            for o in self.obstacles[:]:
                if o['type'] == self.OBSTACLE_TYPE_NRG:
                    obs_rect = pygame.Rect(o['pos'][0] - o['radius'], o['pos'][1] - o['radius'], o['radius'] * 2, o['radius'] * 2)
                    if proj_rect.colliderect(obs_rect):
                        reward += 1.0
                        # Spawn energy particles
                        for _ in range(5):
                            self.particles.append({
                                'pos': list(o['pos']),
                                'life': self.PARTICLE_LIFESPAN
                            })
                        if o in self.obstacles: self.obstacles.remove(o)
                        if p in self.projectiles: self.projectiles.remove(p)
                        # SFX: Obstacle destroyed sound
                        break

        # Player vs Particles
        for p in self.particles[:]:
            part_rect = pygame.Rect(p['pos'][0] - self.PARTICLE_RADIUS, p['pos'][1] - self.PARTICLE_RADIUS, self.PARTICLE_RADIUS * 2, self.PARTICLE_RADIUS * 2)
            if player_rect.colliderect(part_rect):
                self.player_energy = min(self.PLAYER_ENERGY_MAX, self.player_energy + self.PARTICLE_ENERGY_GAIN)
                reward += 0.1
                self.particles.remove(p)
                # SFX: Energy pickup sound
        
        self.player_health = max(0, self.player_health)
        return reward

    def _spawn_obstacles(self):
        if self.np_random.random() < self.obstacle_spawn_prob:
            obstacle_type = self.np_random.choice([self.OBSTACLE_TYPE_DMG, self.OBSTACLE_TYPE_NRG, self.OBSTACLE_TYPE_PHASE])
            speed_multiplier = self.np_random.uniform(0.9, 1.2)
            self.obstacles.append({
                'pos': [self.WIDTH + self.OBSTACLE_RADIUS, self.np_random.integers(self.OBSTACLE_RADIUS, self.HEIGHT - self.OBSTACLE_RADIUS)],
                'radius': self.OBSTACLE_RADIUS,
                'type': obstacle_type,
                'speed': self.obstacle_base_speed * speed_multiplier
            })
            
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_base_speed += 0.25
            self.obstacle_spawn_prob = min(0.06, self.obstacle_spawn_prob + 0.002)

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = self.np_random.integers(-4, 5)
            render_offset[1] = self.np_random.integers(-4, 5)

        self.screen.blit(self.bg_surface, (0, 0))
        self._render_grid(render_offset)
        self._render_game(render_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "energy": self.player_energy}

    def _create_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.WIDTH, y))

    def _render_grid(self, offset):
        for line in self.grid_lines:
            line['pos'] -= self.WORLD_SPEED * (line['width'] / 3.0)
            if line['pos'] < -line['width']:
                line['pos'] = self.WIDTH
            
            x, y_off = int(line['pos'] + offset[0]), int(offset[1])
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, y_off), (x, self.HEIGHT + y_off), line['width'])
            
    def _render_glow(self, surface, color, center, radius, layers=5):
        for i in range(layers, 0, -1):
            alpha = int(80 * (i / layers)**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius * (1 + (layers - i) * 0.15)), glow_color)

    def _render_game(self, offset):
        # Render particles
        for p in self.particles:
            pos = (p['pos'][0] + offset[0], p['pos'][1] + offset[1])
            self._render_glow(self.screen, self.COLOR_PARTICLE, pos, self.PARTICLE_RADIUS, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.PARTICLE_RADIUS, self.COLOR_PARTICLE)

        # Render obstacles
        for o in self.obstacles:
            pos = (o['pos'][0] + offset[0], o['pos'][1] + offset[1])
            if o['type'] == self.OBSTACLE_TYPE_DMG: color = self.COLOR_OBSTACLE_DMG
            elif o['type'] == self.OBSTACLE_TYPE_NRG: color = self.COLOR_OBSTACLE_NRG
            else: color = self.COLOR_OBSTACLE_PHASE
            
            self._render_glow(self.screen, color, pos, o['radius'])
            if o['type'] == self.OBSTACLE_TYPE_DMG: # Square
                rect = pygame.Rect(pos[0] - o['radius'], pos[1] - o['radius'], o['radius']*2, o['radius']*2)
                pygame.draw.rect(self.screen, color, rect)
            else: # Circle
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), o['radius'], color)

        # Render projectiles
        for p in self.projectiles:
            pos = (p['pos'][0] + offset[0], p['pos'][1] + offset[1])
            self._render_glow(self.screen, self.COLOR_PROJECTILE, pos, p['radius'], 3)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), p['radius'], self.COLOR_PROJECTILE)

        # Render player
        player_color = self.COLOR_PLAYER_ATTACK if self.player_state == self.PLAYER_STATE_ATTACK else self.COLOR_PLAYER_DODGE
        player_center = (self.player_pos[0] + offset[0], self.player_pos[1] + offset[1])
        self._render_glow(self.screen, player_color, player_center, self.PLAYER_RADIUS)
        
        # Player triangle shape
        p1 = (player_center[0] + self.PLAYER_RADIUS, player_center[1])
        p2 = (player_center[0] - self.PLAYER_RADIUS / 2, player_center[1] - self.PLAYER_RADIUS)
        p3 = (player_center[0] - self.PLAYER_RADIUS / 2, player_center[1] + self.PLAYER_RADIUS)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), player_color)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), player_color)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health Bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        health_color = (
            self.COLOR_UI_HEALTH_LOW[0] * (1 - health_ratio) + self.COLOR_UI_HEALTH[0] * health_ratio,
            self.COLOR_UI_HEALTH_LOW[1] * (1 - health_ratio) + self.COLOR_UI_HEALTH[1] * health_ratio,
            self.COLOR_UI_HEALTH_LOW[2] * (1 - health_ratio) + self.COLOR_UI_HEALTH[2] * health_ratio,
        )
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (10, 10, 204, 24))
        pygame.draw.rect(self.screen, self.COLOR_BG_START, (12, 12, 200, 20))
        if health_ratio > 0: pygame.draw.rect(self.screen, health_color, (12, 12, 200 * health_ratio, 20))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (14, 14))

        # Energy Bar
        energy_ratio = self.player_energy / self.PLAYER_ENERGY_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (10, 38, 204, 14))
        pygame.draw.rect(self.screen, self.COLOR_BG_START, (12, 40, 200, 10))
        if energy_ratio > 0: pygame.draw.rect(self.screen, self.COLOR_UI_ENERGY, (12, 40, 200 * energy_ratio, 10))
        
        # Score and Progress
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))
        
        progress_percent = int((self.steps / self.MAX_STEPS) * 100)
        progress_text = self.font_small.render(f"PROGRESS: {progress_percent}%", True, self.COLOR_WHITE)
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 15, 45))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv()
    
    # --- Manual Play ---
    # Use Arrow Keys to move (though unused in this game), Space to shoot, Left Shift to switch state
    # Pygame window setup for human play
    # Re-enable video driver for local play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Neon Runner")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # no-op, released, released
        # actions[0] (movement) is ignored by the game logic
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1 # Shift held
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over! Final Info: {info}")
    env.close()