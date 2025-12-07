import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:08:05.438564
# Source Brief: brief_02365.md
# Brief Index: 2365
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
        "Defend your central base from waves of attacking viruses. Use your primary and secondary weapons to survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to fire your primary weapon and shift to fire the powerful piercer shot."
    )
    auto_advance = True

    # --- Constants ---
    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2500
    TOTAL_WAVES = 20
    WAVE_TRANSITION_TIME = 90  # frames (3 seconds at 30fps)

    # Colors
    COLOR_BG = (10, 20, 35)
    COLOR_BASE = (50, 200, 100)
    COLOR_BASE_GLOW = (100, 255, 150)
    COLOR_VIRUS = (220, 50, 50)
    COLOR_VIRUS_GLOW = (255, 100, 100)
    COLOR_PROJECTILE_PRIMARY = (100, 150, 255)
    COLOR_PROJECTILE_SECONDARY = (255, 200, 50)
    COLOR_RETICLE = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 40, 70, 180) # RGBA

    # Player/Base settings
    BASE_INITIAL_HEALTH = 100
    BASE_INITIAL_RADIUS = 60
    BASE_SHRINK_RATE = 0.01  # Radius loss per step
    RETICLE_SPEED = 8

    # Weapon settings
    PRIMARY_COOLDOWN = 10 # frames
    SECONDARY_COOLDOWN = 30 # frames
    SECONDARY_UNLOCK_WAVE = 5
    SECONDARY_AMMO_GAIN = 3 # ammo per virus kill
    PROJECTILE_SPEED = 12
    PROJECTILE_RADIUS = 5
    SECONDARY_PROJECTILE_RADIUS = 8

    # Virus settings
    VIRUS_BASE_SPEED = 0.8
    VIRUS_SPEED_INC_PER_WAVE = 0.05
    VIRUS_BASE_HEALTH = 1
    VIRUS_HEALTH_INC_WAVE_INTERVAL = 5
    VIRUS_RADIUS = 12

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self.steps = 0
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        self.wave_transition_timer = 0

        # Base state
        self.base_health = self.BASE_INITIAL_HEALTH
        self.base_radius = self.BASE_INITIAL_RADIUS
        self.base_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        # Player state
        self.reticle_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.prev_space_held = False
        self.prev_shift_held = False
        self.primary_cooldown_timer = 0
        self.secondary_cooldown_timer = 0
        self.secondary_unlocked = False
        self.secondary_ammo = 0
        
        # Entity lists
        self.projectiles = []
        self.viruses = []
        self.particles = []
        self.damage_events = []

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Handle wave transitions
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self.wave += 1
                if self.wave > self.TOTAL_WAVES:
                    self.game_over = True # Victory
                else:
                    self._start_new_wave()
        
        # Update cooldowns
        if self.primary_cooldown_timer > 0: self.primary_cooldown_timer -= 1
        if self.secondary_cooldown_timer > 0: self.secondary_cooldown_timer -= 1

        # Process actions
        self._handle_input(action)

        # Update game objects
        reward += self._update_projectiles()
        reward += self._update_viruses()
        self._update_base()
        self._update_particles()

        # Check for wave completion
        if not self.viruses and self.wave_transition_timer == 0 and self.wave <= self.TOTAL_WAVES:
            reward += 5.0  # Wave clear reward
            self.wave_transition_timer = self.WAVE_TRANSITION_TIME

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.wave > self.TOTAL_WAVES:
                reward += 100.0 # Victory bonus
            else:
                reward -= 100.0 # Defeat penalty
        
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Reticle movement
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED
        
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.SCREEN_WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.SCREEN_HEIGHT)

        # Primary weapon fire (on press)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and self.primary_cooldown_timer == 0:
            self._fire_projectile(primary=True)
            self.primary_cooldown_timer = self.PRIMARY_COOLDOWN
            # sfx: player_shoot_primary.wav

        # Secondary weapon fire (on press)
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.secondary_cooldown_timer == 0 and self.secondary_unlocked and self.secondary_ammo > 0:
            self._fire_projectile(primary=False)
            self.secondary_cooldown_timer = self.SECONDARY_COOLDOWN
            self.secondary_ammo -= 1
            # sfx: player_shoot_secondary.wav
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _fire_projectile(self, primary):
        direction = self.reticle_pos - self.base_center
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        
        projectile = {
            'pos': np.array(self.base_center, dtype=float),
            'vel': direction * self.PROJECTILE_SPEED,
            'primary': primary,
            'pierced': 0
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            
            hit_something = False
            for v in self.viruses:
                dist = np.linalg.norm(p['pos'] - v['pos'])
                if dist < (self.PROJECTILE_RADIUS if p['primary'] else self.SECONDARY_PROJECTILE_RADIUS) + self.VIRUS_RADIUS:
                    hit_something = True
                    v['health'] -= 1
                    reward += 0.1 # Hit reward
                    self._create_explosion(p['pos'], 5, self.COLOR_PROJECTILE_PRIMARY if p['primary'] else self.COLOR_PROJECTILE_SECONDARY, 0.5)

                    if v['health'] <= 0:
                        reward += 1.0 # Kill reward
                        self.score += 10
                        if self.secondary_unlocked:
                            self.secondary_ammo += self.SECONDARY_AMMO_GAIN
                        v['active'] = False
                        self._create_explosion(v['pos'], 20, self.COLOR_VIRUS_GLOW, 1.0)
                        # sfx: virus_die.wav
                    else:
                        # sfx: virus_hit.wav
                        pass

                    if p['primary']:
                        break # Primary projectiles are destroyed on hit
                    else: # Secondary can pierce
                        p['pierced'] += 1
                        if p['pierced'] >= 3:
                           break

            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT):
                hit_something = True # Mark for removal if it goes off-screen

            if not hit_something or not p['primary']:
                 projectiles_to_keep.append(p)

        self.projectiles = [p for p in projectiles_to_keep if not (p.get('pierced', 0) >= 3 and not p['primary'])]
        self.viruses = [v for v in self.viruses if v.get('active', True)]
        return reward

    def _update_viruses(self):
        reward = 0
        base_damaged_this_step = False
        for v in self.viruses:
            direction = self.base_center - v['pos']
            distance = np.linalg.norm(direction)
            if distance > 1:
                direction /= distance

            # Add perpendicular sinusoidal motion for variety
            perp_dir = np.array([-direction[1], direction[0]])
            v['pos'] += direction * v['speed'] + perp_dir * math.sin(self.steps * v['wobble_freq'] + v['wobble_phase']) * 0.5

            # Check collision with base
            if distance < self.base_radius + self.VIRUS_RADIUS:
                self.base_health -= v['damage']
                base_damaged_this_step = True
                v['active'] = False
                self._create_explosion(v['pos'], 15, self.COLOR_BASE_GLOW, 0.8)
                self.damage_events.append({'pos': v['pos'], 'timer': 20})
                # sfx: base_damage.wav
        
        if base_damaged_this_step:
            reward -= 0.1

        self.viruses = [v for v in self.viruses if v.get('active', True)]
        return reward

    def _update_base(self):
        self.base_radius = max(10, self.BASE_INITIAL_RADIUS - self.steps * self.BASE_SHRINK_RATE)
        self.base_health = max(0, self.base_health)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95
        
        self.damage_events = [d for d in self.damage_events if d['timer'] > 0]
        for d in self.damage_events:
            d['timer'] -=1

    def _start_new_wave(self):
        if self.wave == self.SECONDARY_UNLOCK_WAVE:
            self.secondary_unlocked = True

        num_viruses = 3 + self.wave
        virus_speed = self.VIRUS_BASE_SPEED + self.VIRUS_SPEED_INC_PER_WAVE * (self.wave - 1)
        virus_health = self.VIRUS_BASE_HEALTH + (self.wave -1) // self.VIRUS_HEALTH_INC_WAVE_INTERVAL

        for _ in range(num_viruses):
            # Spawn on edge of screen
            edge = self.np_random.integers(0, 4)
            if edge == 0: # top
                pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -self.VIRUS_RADIUS]
            elif edge == 1: # bottom
                pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.VIRUS_RADIUS]
            elif edge == 2: # left
                pos = [-self.VIRUS_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            else: # right
                pos = [self.SCREEN_WIDTH + self.VIRUS_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            
            self.viruses.append({
                'pos': np.array(pos, dtype=float),
                'speed': virus_speed * self.np_random.uniform(0.8, 1.2),
                'health': virus_health,
                'damage': 10,
                'active': True,
                'wobble_freq': self.np_random.uniform(0.05, 0.15),
                'wobble_phase': self.np_random.uniform(0, 2 * math.pi)
            })

    def _create_explosion(self, pos, num_particles, color, speed_mult):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.np_random.integers(15, 31),
                'radius': self.np_random.uniform(1, 4),
                'color': color
            })

    def _check_termination(self):
        return self.base_health <= 0 or (self.wave > self.TOTAL_WAVES and not self.viruses)

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
            "wave": self.wave,
            "base_health": self.base_health,
            "secondary_ammo": self.secondary_ammo,
        }
    
    def _render_game(self):
        # Base
        base_pos = (int(self.base_center[0]), int(self.base_center[1]))
        pygame.gfxdraw.filled_circle(self.screen, base_pos[0], base_pos[1], int(self.base_radius), self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_pos[0], base_pos[1], int(self.base_radius), self.COLOR_BASE_GLOW)
        
        # Base damage flash
        for d_event in self.damage_events:
            alpha = int(255 * (d_event['timer'] / 20))
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(flash_surface, (255, 50, 50, alpha), base_pos, int(self.base_radius) + 10)
            self.screen.blit(flash_surface, (0,0))

        # Particles
        for p in self.particles:
            alpha = int(255 * p['life'] / 30)
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(max(0, p['radius'])), color)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            prev_pos = (int(p['pos'][0] - p['vel'][0]), int(p['pos'][1] - p['vel'][1]))
            color = self.COLOR_PROJECTILE_PRIMARY if p['primary'] else self.COLOR_PROJECTILE_SECONDARY
            radius = self.PROJECTILE_RADIUS if p['primary'] else self.SECONDARY_PROJECTILE_RADIUS
            pygame.draw.line(self.screen, color, prev_pos, pos, radius * 2)
            pygame.draw.circle(self.screen, (255,255,255), pos, radius)

        # Viruses
        for v in self.viruses:
            pos = (int(v['pos'][0]), int(v['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.VIRUS_RADIUS, self.COLOR_VIRUS)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.VIRUS_RADIUS, self.COLOR_VIRUS_GLOW)
        
        # Reticle
        ret_pos = (int(self.reticle_pos[0]), int(self.reticle_pos[1]))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_pos[0] - 10, ret_pos[1]), (ret_pos[0] + 10, ret_pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (ret_pos[0], ret_pos[1] - 10), (ret_pos[0], ret_pos[1] + 10), 2)
        
    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health
        health_text = self.font_small.render(f"HEALTH: {int(self.base_health)}%", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (180, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 150, 10))
        
        # Ammo
        if self.secondary_unlocked:
            ammo_color = self.COLOR_PROJECTILE_SECONDARY if self.secondary_ammo > 0 else self.COLOR_VIRUS
            ammo_text = self.font_small.render(f"PIERCER AMMO: {self.secondary_ammo}", True, ammo_color)
            self.screen.blit(ammo_text, (330, 10))

        # Wave transition text
        if self.wave_transition_timer > 0:
            msg = f"WAVE {self.wave} COMPLETE" if self.wave_transition_timer > 30 else f"WAVE {self.wave + 1} INCOMING"
            wave_msg = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = wave_msg.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(wave_msg, text_rect)
            
        # Game Over Text
        if self.game_over:
            msg = "VICTORY" if self.wave > self.TOTAL_WAVES else "BASE DESTROYED"
            color = self.COLOR_BASE_GLOW if self.wave > self.TOTAL_WAVES else self.COLOR_VIRUS_GLOW
            end_msg = self.font_large.render(msg, True, color)
            text_rect = end_msg.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_msg, text_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for interactive testing and will not be run by the evaluation system.
    # It requires a graphical display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="human")
    
    # Simple interactive loop for testing
    obs, info = env.reset()
    done = False
    
    # Override the screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cellular Defense")

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        rendered_frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rendered_frame)
        env.screen.blit(surf, (0,0))
        pygame.display.flip()

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()