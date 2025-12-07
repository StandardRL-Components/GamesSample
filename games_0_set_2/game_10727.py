import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:58:13.896747
# Source Brief: brief_00727.md
# Brief Index: 727
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive as a shrinking cell by collecting resources to craft a protective shell and avoiding deadly predators."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Collect resources and press space to craft a protective shell."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG_START = (5, 0, 20)
    COLOR_BG_END = (0, 0, 0)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 40)
    COLOR_PREDATOR = (255, 50, 50)
    COLOR_PREDATOR_GLOW = (255, 50, 50, 60)
    COLOR_OBSTACLE = (100, 100, 110)
    COLOR_POWERUP = (50, 150, 255)
    COLOR_POWERUP_GLOW = (50, 150, 255, 80)
    COLOR_SHELL = (255, 255, 0, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 200, 150)

    # Player
    PLAYER_START_SIZE = 15.0
    PLAYER_MAX_SPEED = 6.0
    PLAYER_ACCELERATION = 0.8
    PLAYER_DRAG = 0.92
    
    # Game Mechanics
    SHRINK_RATE = 0.01
    SHRINK_BOOST_RATE = 0.05
    SHRINK_BOOST_DURATION = 90 # frames
    SHELL_COST = 3
    
    # Procedural Generation
    INITIAL_PREDATORS = 1
    INITIAL_OBSTACLES = 5
    INITIAL_POWERUPS = 5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_size = 0.0
        self.player_shrink_rate = 0.0
        self.shrink_boost_timer = 0
        
        self.resources = 0
        self.shell_active = False
        self.prev_space_held = False

        self.predators = []
        self.obstacles = []
        self.powerups = []
        self.particles = []

        self.predator_spawn_counter = 0
        self.predator_base_speed = 1.0
        self.predator_spawn_prob = 0.01


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_size = self.PLAYER_START_SIZE
        self.player_shrink_rate = self.SHRINK_RATE
        self.shrink_boost_timer = 0

        # Crafting
        self.resources = 0
        self.shell_active = False
        self.prev_space_held = False

        # Entities
        self.predators = []
        self.obstacles = []
        self.powerups = []
        self.particles = []
        
        # Difficulty
        self.predator_base_speed = 1.0
        self.predator_spawn_prob = 0.01

        # --- Procedural Generation ---
        occupied_areas = [(self.player_pos, self.PLAYER_START_SIZE * 4)]

        for _ in range(self.INITIAL_OBSTACLES):
            self._spawn_entity(self.obstacles, 20, 40, occupied_areas)
        for _ in range(self.INITIAL_POWERUPS):
            self._spawn_entity(self.powerups, 8, 8, occupied_areas)
        for _ in range(self.INITIAL_PREDATORS):
            self._spawn_predator(occupied_areas)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input & Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        accel_vec = np.zeros(2, dtype=np.float32)
        if movement == 1: accel_vec[1] = -1 # Up
        elif movement == 2: accel_vec[1] = 1  # Down
        elif movement == 3: accel_vec[0] = -1 # Left
        elif movement == 4: accel_vec[0] = 1  # Right
        
        self.player_vel += accel_vec * self.PLAYER_ACCELERATION
        
        # Crafting (on button press)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and self.resources >= self.SHELL_COST and not self.shell_active:
            self.shell_active = True
            self.resources -= self.SHELL_COST
            reward += 5.0
            # SFX: Crafting success
            self._create_particles(self.player_pos, 30, (255, 255, 0), 2.0, 0.5)

        self.prev_space_held = space_held

        # --- 2. Update Game Logic ---
        self.steps += 1
        self.score = self.steps # Score is survival time
        reward += 0.1 # Survival reward

        # Player update
        self.player_vel *= self.PLAYER_DRAG
        if np.linalg.norm(self.player_vel) > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel / np.linalg.norm(self.player_vel) * self.PLAYER_MAX_SPEED
        self.player_pos += self.player_vel
        self._wrap_around(self.player_pos)
        
        # Shrinking
        if self.shrink_boost_timer > 0:
            self.player_shrink_rate = self.SHRINK_BOOST_RATE
            self.shrink_boost_timer -= 1
        else:
            self.player_shrink_rate = self.SHRINK_RATE
        self.player_size = max(0, self.player_size - self.player_shrink_rate)
        
        # Predator update
        for predator in self.predators:
            self._update_predator(predator)
        
        # Particle update
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95

        # --- 3. Handle Collisions ---
        # Player vs Obstacles
        for obs in self.obstacles:
            if self._check_collision(self.player_pos, self.player_size, obs['pos'], obs['size']):
                # Simple momentum-based bounce
                overlap = (self.player_size + obs['size']) - np.linalg.norm(self.player_pos - obs['pos'])
                if np.linalg.norm(self.player_pos - obs['pos']) > 1e-6:
                    normal = (self.player_pos - obs['pos']) / np.linalg.norm(self.player_pos - obs['pos'])
                    self.player_pos += normal * overlap
                    self.player_vel -= 2 * np.dot(self.player_vel, normal) * normal * 0.5 # Bounce with energy loss
                # SFX: Thud
        
        # Player vs Powerups
        for pow in self.powerups[:]:
            if self._check_collision(self.player_pos, self.player_size, pow['pos'], pow['size']):
                self.resources += 1
                reward += 1.0
                self.powerups.remove(pow)
                self._spawn_entity(self.powerups, 8, 8, []) # Spawn a new one
                # SFX: Powerup collect
                self._create_particles(pow['pos'], 15, self.COLOR_POWERUP, 1.5, 0.8)

        # Player vs Predators
        for pred in self.predators:
            if self._check_collision(self.player_pos, self.player_size, pred['pos'], pred['size']):
                if self.shell_active:
                    self.shell_active = False
                    self.shrink_boost_timer = self.SHRINK_BOOST_DURATION
                    # Knockback
                    if np.linalg.norm(self.player_pos - pred['pos']) > 1e-6:
                        knockback_dir = (self.player_pos - pred['pos']) / np.linalg.norm(self.player_pos - pred['pos'])
                        self.player_vel = knockback_dir * self.PLAYER_MAX_SPEED * 1.5
                    pred['vel'] *= -0.5 # Stun predator briefly
                    # SFX: Shell break
                    self._create_particles(self.player_pos, 50, (255, 255, 0), 3.0, 1.0)
                else:
                    self.game_over = True
                    reward = -100.0
                    # SFX: Player death
                    self._create_particles(self.player_pos, 100, self.COLOR_PLAYER, 4.0, 0.2)
                break

        # --- 4. Difficulty Scaling & Spawning ---
        if self.steps % 100 == 0:
            self.predator_spawn_prob = min(0.1, self.predator_spawn_prob + 0.001)
        if self.steps % 500 == 0:
            self.predator_base_speed = min(3.0, self.predator_base_speed + 0.1)

        if self.np_random.random() < self.predator_spawn_prob:
            self._spawn_predator([])

        # --- 5. Check Termination ---
        terminated = self.game_over
        truncated = False
        if self.player_size <= 0:
            if not terminated: reward = -100.0
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            if not terminated: reward = 100.0
            truncated = True # Use truncated for time limits
            terminated = True # Gymnasium standard is to set both true
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources}
    
    # --- Rendering Helpers ---
    def _render_background(self):
        self.screen.fill(self.COLOR_BG_END)
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Obstacles
        for obs in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(obs['pos'][0]), int(obs['pos'][1]), int(obs['size']), self.COLOR_OBSTACLE)
        
        # Powerups
        for pow in self.powerups:
            self._draw_glowing_circle(pow['pos'], pow['size'], self.COLOR_POWERUP, self.COLOR_POWERUP_GLOW)
        
        # Predators
        for pred in self.predators:
            self._draw_glowing_circle(pred['pos'], pred['size'], self.COLOR_PREDATOR, self.COLOR_PREDATOR_GLOW)
            
        # Shell
        if self.shell_active:
            shell_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(shell_surf, int(self.player_pos[0]), int(self.player_pos[1]), int(self.player_size + 5), self.COLOR_SHELL)
            pygame.gfxdraw.filled_circle(shell_surf, int(self.player_pos[0]), int(self.player_pos[1]), int(self.player_size + 5), self.COLOR_SHELL)
            self.screen.blit(shell_surf, (0, 0))

        # Player
        if self.player_size > 0:
            self._draw_glowing_circle(self.player_pos, self.player_size, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((int(p['size'])*2, int(p['size'])*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_ui(self):
        # Player Size
        size_text = f"SIZE: {self.player_size:.1f}"
        size_surf = self.font_large.render(size_text, True, self.COLOR_TEXT)
        self.screen.blit(size_surf, (10, 10))
        
        # Survival Time
        time_text = f"TIME: {self.steps}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Resources
        res_text = self.font_small.render("RESOURCES:", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 45))
        for i in range(self.resources):
            color = self.COLOR_POWERUP if i < self.SHELL_COST else (100, 100, 150)
            pygame.draw.rect(self.screen, color, (110 + i * 15, 45, 12, 12))
        
        # Crafting Cost
        cost_text = self.font_small.render(f"(COST: {self.SHELL_COST})", True, self.COLOR_TEXT)
        self.screen.blit(cost_text, (10, 65))

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        if radius <= 0: return
        x, y = int(pos[0]), int(pos[1])
        rad = int(radius)
        
        # Glow
        glow_surf = pygame.Surface((rad * 4, rad * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, rad * 2, rad * 2, int(rad * 1.5), glow_color)
        self.screen.blit(glow_surf, (x - rad * 2, y - rad * 2), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main circle
        pygame.gfxdraw.aacircle(self.screen, x, y, rad, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, rad, color)

    # --- Game Logic Helpers ---
    def _spawn_entity(self, entity_list, min_size, max_size, occupied_areas):
        for _ in range(100): # Max 100 attempts to find a free spot
            size = self.np_random.uniform(min_size, max_size)
            pos = np.array([self.np_random.uniform(size, self.WIDTH - size),
                            self.np_random.uniform(size, self.HEIGHT - size)], dtype=np.float32)
            
            is_overlapping = False
            for occ_pos, occ_size in occupied_areas:
                if np.linalg.norm(pos - occ_pos) < size + occ_size:
                    is_overlapping = True
                    break
            if not is_overlapping:
                entity_list.append({'pos': pos, 'size': size})
                occupied_areas.append((pos, size))
                return
    
    def _spawn_predator(self, occupied_areas):
        for _ in range(100):
            size = self.np_random.uniform(10, 20)
            pos = np.array([self.np_random.uniform(size, self.WIDTH - size),
                            self.np_random.uniform(size, self.HEIGHT - size)], dtype=np.float32)

            is_overlapping = False
            for occ_pos, occ_size in occupied_areas:
                if np.linalg.norm(pos - occ_pos) < size + occ_size:
                    is_overlapping = True
                    break
            if not is_overlapping:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.predator_base_speed * self.np_random.uniform(0.8, 1.2)
                
                pred_type = 'circular' if self.np_random.random() > 0.5 else 'linear'
                if pred_type == 'circular':
                    vel = np.zeros(2)
                else:
                    vel = np.array([math.cos(angle), math.sin(angle)]) * speed

                predator = {
                    'pos': pos, 'size': size, 'vel': vel, 'speed': speed,
                    'type': pred_type, 'angle': angle,
                    'center': pos.copy(), 'radius': self.np_random.uniform(30, 80)
                }
                self.predators.append(predator)
                occupied_areas.append((pos, size))
                return

    def _update_predator(self, predator):
        if predator['type'] == 'linear':
            predator['pos'] += predator['vel']
            if predator['pos'][0] < 0 or predator['pos'][0] > self.WIDTH: predator['vel'][0] *= -1
            if predator['pos'][1] < 0 or predator['pos'][1] > self.HEIGHT: predator['vel'][1] *= -1
        elif predator['type'] == 'circular':
            predator['angle'] += predator['speed'] * 0.05
            predator['pos'][0] = predator['center'][0] + math.cos(predator['angle']) * predator['radius']
            predator['pos'][1] = predator['center'][1] + math.sin(predator['angle']) * predator['radius']
        self._wrap_around(predator['pos'])

    def _check_collision(self, pos1, r1, pos2, r2):
        if r1 <= 0 or r2 <= 0: return False
        return np.linalg.norm(pos1 - pos2) < r1 + r2

    def _wrap_around(self, pos):
        pos[0] %= self.WIDTH
        pos[1] %= self.HEIGHT
        
    def _create_particles(self, pos, count, color, speed_mult, spread):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed + self.np_random.uniform(-spread, spread, 2)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(2, 5),
                'life': life,
                'max_life': life,
                'color': color
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()
        

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    # --- Manual Play ---
    pygame.display.set_caption("Shrinking Cell Survival")
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # None
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Blit the observation from the environment to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if done:
            print(f"Game Over! Final Score (Steps): {info['score']}")
            # Wait a bit before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()