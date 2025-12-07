import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:37:33.421685
# Source Brief: brief_02899.md
# Brief Index: 2899
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
        "Survive as long as possible by balancing your health, energy, and sanity. "
        "Dodge incoming threats while performing actions to replenish your dwindling resources."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to perform actions that balance your energy and sanity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (16, 16, 24)  # Dark blue-gray
    COLOR_HEALTH = (76, 175, 80)
    COLOR_ENERGY = (33, 150, 243)
    COLOR_SANITY = (156, 39, 176)
    COLOR_ENEMY = (244, 67, 54)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 40, 50)
    COLOR_UI_BORDER = (60, 60, 70)

    # Game Parameters
    NATURAL_DEPLETION = {
        'health': 0.02,
        'energy': 0.15,
        'sanity': 0.12
    }
    ACTION_EFFECTS = {
        'left': {'energy': 2.0, 'sanity': -1.0, 'health': -0.5},
        'right': {'sanity': 2.0, 'energy': -1.0, 'health': -0.5}
    }
    ENEMY_SPAWN_RATE = 45  # Lower is more frequent
    ENEMY_BASE_SPEED = 1.5
    ENEMY_SPEED_INCREASE_INTERVAL = 300  # 5 seconds
    ENEMY_SPEED_INCREASE_AMOUNT = 0.1
    ENEMY_SIZE = 12
    DANGER_ZONE_HEIGHT = 100

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 28)

        self.render_mode = render_mode
        
        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.health = 0.0
        self.energy = 0.0
        self.sanity = 0.0
        self.visual_health = 0.0
        self.visual_energy = 0.0
        self.visual_sanity = 0.0
        self.enemies = []
        self.particles = []
        self.starfield = []
        self.enemy_spawn_timer = 0
        self.current_enemy_speed = self.ENEMY_BASE_SPEED
        self.last_action_feedback = {'type': None, 'timer': 0}
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.health = 100.0
        self.energy = 100.0
        self.sanity = 100.0
        self.visual_health = 100.0
        self.visual_energy = 100.0
        self.visual_sanity = 100.0

        self.enemies = []
        self.particles = []
        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE
        self.current_enemy_speed = self.ENEMY_BASE_SPEED
        self.last_action_feedback = {'type': None, 'timer': 0}

        if not self.starfield:
            for _ in range(150):
                self.starfield.append({
                    'pos': [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)],
                    'size': random.uniform(0.5, 1.5),
                    'speed': random.uniform(0.1, 0.3)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_actions(action)
        self._update_game_state()
        
        reward += self._calculate_reward()
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and self.win:
            reward += 100.0 # Win bonus
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement = action[0]
        
        action_taken = None
        if movement == 3: # Left
            action_taken = 'left'
        elif movement == 4: # Right
            action_taken = 'right'
        
        if action_taken:
            effects = self.ACTION_EFFECTS[action_taken]
            self.health += effects['health']
            self.energy += effects['energy']
            self.sanity += effects['sanity']
            # Sound placeholder: sfx_resource_gain.wav
            self.last_action_feedback = {'type': action_taken, 'timer': 15}
            if action_taken == 'left':
                self._create_particle_burst(pygame.Vector2(160, 55), self.COLOR_ENERGY, 20)
            else: # right
                self._create_particle_burst(pygame.Vector2(480, 55), self.COLOR_SANITY, 20)


    def _update_game_state(self):
        # Natural resource depletion
        self.health -= self.NATURAL_DEPLETION['health']
        self.energy -= self.NATURAL_DEPLETION['energy']
        self.sanity -= self.NATURAL_DEPLETION['sanity']

        # Clamp resources
        self.health = max(0, min(100, self.health))
        self.energy = max(0, min(100, self.energy))
        self.sanity = max(0, min(100, self.sanity))

        # Update visual bars smoothly
        lerp_factor = 0.1
        self.visual_health += (self.health - self.visual_health) * lerp_factor
        self.visual_energy += (self.energy - self.visual_energy) * lerp_factor
        self.visual_sanity += (self.sanity - self.visual_sanity) * lerp_factor

        # Update enemies
        danger_zone = pygame.Rect(0, 0, self.WIDTH, self.DANGER_ZONE_HEIGHT)
        for enemy in self.enemies[:]:
            enemy['pos'] += enemy['vel']
            enemy['rect'].center = enemy['pos']

            # Enemy collision with danger zone
            if enemy['rect'].colliderect(danger_zone):
                self.health -= 5.0
                # Sound placeholder: sfx_player_hit.wav
                self._create_particle_burst(enemy['pos'], self.COLOR_HEALTH, 15, speed_mult=0.5)
                self.enemies.remove(enemy)
                continue

            # Remove off-screen enemies
            if not self.screen.get_rect().inflate(50, 50).colliderect(enemy['rect']):
                self.enemies.remove(enemy)
            else:
                # Enemy trail particles
                if self.steps % 3 == 0:
                    self.particles.append(self._create_particle(enemy['pos'], self.COLOR_ENEMY, 2, 15, -enemy['vel'] * 0.1))

        # Spawn new enemies
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE

        # Update difficulty
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            self.current_enemy_speed += self.ENEMY_SPEED_INCREASE_AMOUNT

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
        
        # Update starfield
        for star in self.starfield:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][1] = 0
                star['pos'][0] = random.uniform(0, self.WIDTH)
        
        # Update action feedback timer
        if self.last_action_feedback['timer'] > 0:
            self.last_action_feedback['timer'] -= 1
        else:
            self.last_action_feedback['type'] = None


    def _spawn_enemy(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), -self.ENEMY_SIZE)
            angle = random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 'bottom':
            pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_SIZE)
            angle = random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 'left':
            pos = pygame.Vector2(-self.ENEMY_SIZE, random.uniform(self.DANGER_ZONE_HEIGHT, self.HEIGHT))
            angle = random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else: # right
            pos = pygame.Vector2(self.WIDTH + self.ENEMY_SIZE, random.uniform(self.DANGER_ZONE_HEIGHT, self.HEIGHT))
            angle = random.uniform(math.pi * 0.75, math.pi * 1.25)
        
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.current_enemy_speed
        
        self.enemies.append({
            'pos': pos,
            'vel': vel,
            'rect': pygame.Rect(0, 0, self.ENEMY_SIZE, self.ENEMY_SIZE)
        })

    def _calculate_reward(self):
        # Survival reward
        reward = 0.1
        
        # Penalty for enemy collision (applied in _update_game_state via health loss)
        # Check health change to infer collision
        if self.health < self.visual_health - 4.0: # Check if health dropped significantly
             reward -= 5.0

        return reward

    def _check_termination(self):
        if self.health <= 0 or self.energy <= 0 or self.sanity <= 0:
            self.game_over = True
            # Sound placeholder: sfx_game_over.wav
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            # Sound placeholder: sfx_win.wav
            return True
        return False

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
            "health": self.health,
            "energy": self.energy,
            "sanity": self.sanity,
        }

    def _render_game(self):
        # Render starfield
        for star in self.starfield:
            size = int(star['size'])
            color_val = int(star['speed'] * 100 + 50)
            color = (color_val, color_val, color_val)
            pygame.draw.rect(self.screen, color, (int(star['pos'][0]), int(star['pos'][1]), size, size))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            self._draw_glowing_circle(self.screen, p['pos'], p['radius'], color)

        # Render enemies
        for enemy in self.enemies:
            self._draw_glowing_rect(self.screen, enemy['rect'], self.COLOR_ENEMY, 5)

    def _render_ui(self):
        # Draw UI background
        ui_rect = pygame.Rect(0, 0, self.WIDTH, self.DANGER_ZONE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, (0, self.DANGER_ZONE_HEIGHT), (self.WIDTH, self.DANGER_ZONE_HEIGHT), 2)

        # Resource Bars
        self._draw_resource_bar("HEALTH", self.visual_health, self.COLOR_HEALTH, pygame.Rect(10, 20, 200, 20))
        self._draw_resource_bar("ENERGY", self.visual_energy, self.COLOR_ENERGY, pygame.Rect(10, 55, 200, 20))
        self._draw_resource_bar("SANITY", self.visual_sanity, self.COLOR_SANITY, pygame.Rect(self.WIDTH - 210, 55, 200, 20))
        
        # Action feedback indicators
        if self.last_action_feedback['type'] == 'left':
            pygame.draw.rect(self.screen, self.COLOR_ENERGY, (10, 55, 200, 20), 3)
        elif self.last_action_feedback['type'] == 'right':
            pygame.draw.rect(self.screen, self.COLOR_SANITY, (self.WIDTH - 210, 55, 200, 20), 3)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"{time_left:.1f}"
        text_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, 40))
        self.screen.blit(text_surf, text_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY" if self.win else "ASSIMILATED"
            color = self.COLOR_HEALTH if self.win else self.COLOR_ENEMY
            
            end_text_surf = self.font_large.render(message, True, color)
            end_text_rect = end_text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def _draw_resource_bar(self, label, value, color, rect):
        # Draw bar background
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, rect, 2, border_radius=4)
        
        # Draw filled portion
        fill_width = max(0, (value / 100) * (rect.width - 4))
        fill_rect = pygame.Rect(rect.left + 2, rect.top + 2, fill_width, rect.height - 4)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=2)

        # Draw label
        label_surf = self.font_small.render(label, True, self.COLOR_TEXT)
        label_rect = label_surf.get_rect(midright=(rect.left - 10, rect.centery))
        # self.screen.blit(label_surf, label_rect) # Removed for cleaner look

        # Draw value text
        value_text = f"{int(value)}%"
        value_surf = self.font_small.render(value_text, True, self.COLOR_TEXT)
        value_rect = value_surf.get_rect(midleft=(rect.right + 10, rect.centery))
        self.screen.blit(value_surf, value_rect)

    def _create_particle(self, pos, color, radius, lifespan, vel):
        return {
            'pos': pygame.Vector2(pos),
            'vel': vel,
            'color': color,
            'radius': radius,
            'lifespan': lifespan,
            'max_lifespan': lifespan
        }

    def _create_particle_burst(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = random.uniform(2, 4)
            lifespan = random.randint(15, 30)
            self.particles.append(self._create_particle(pos, color, radius, lifespan, vel))

    def _draw_glowing_circle(self, surface, pos, radius, color):
        temp_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
        center = (radius * 2, radius * 2)
        
        # Glow layers
        for i in range(3):
            alpha = color[3] * (0.5 - i * 0.15)
            if alpha < 0: continue
            pygame.gfxdraw.filled_circle(
                temp_surf, int(center[0]), int(center[1]), 
                int(radius * (1 + i * 0.5)),
                (color[0], color[1], color[2], int(alpha))
            )
        
        # Core circle
        pygame.gfxdraw.filled_circle(temp_surf, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.aacircle(temp_surf, int(center[0]), int(center[1]), int(radius), color)
        
        surface.blit(temp_surf, (int(pos.x - center[0]), int(pos.y - center[1])))

    def _draw_glowing_rect(self, surface, rect, color, glow_size):
        temp_surf = pygame.Surface((rect.width + glow_size * 2, rect.height + glow_size * 2), pygame.SRCALPHA)
        center_rect = pygame.Rect(glow_size, glow_size, rect.width, rect.height)

        # Glow layers
        for i in range(glow_size // 2):
            alpha = 100 - i * 20
            if alpha < 0: continue
            pygame.draw.rect(temp_surf, color + (alpha,), center_rect.inflate(i * 2, i * 2), border_radius=3)
        
        # Core rect
        pygame.draw.rect(temp_surf, color, center_rect, border_radius=3)
        
        surface.blit(temp_surf, (rect.left - glow_size, rect.top - glow_size))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Set a non-dummy driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Resource Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The other actions are not used by human player in this design
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)
        
    env.close()