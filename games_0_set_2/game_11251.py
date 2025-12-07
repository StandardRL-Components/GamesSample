import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:52:54.923333
# Source Brief: brief_01251.md
# Brief Index: 1251
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    game_description = (
        "Defend your system's core in this cyberpunk rhythm game. Match enemy attack patterns and unleash powerful abilities to survive."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to match incoming enemy rhythms. Press space to release an offensive pulse and shift to activate your shield."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CENTER_X, self.CENTER_Y = self.WIDTH // 2, self.HEIGHT // 2
        self.MAX_STEPS = 1000
        self.MAX_HEALTH = 100
        
        # --- Colors (Cyberpunk Neon) ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_GRID = (20, 40, 60)
        self.COLOR_HEART = (0, 255, 255) # Cyan
        self.COLOR_HEALTH_ARC = (0, 255, 128) # Green
        self.COLOR_DAMAGE_ARC = (255, 0, 100) # Red
        self.COLOR_OFFENSE = (255, 255, 0) # Yellow
        self.COLOR_DEFENSE = (0, 128, 255) # Blue
        self.COLOR_GLITCH = (255, 100, 0) # Orange
        self.COLOR_VIRUS = (200, 0, 255) # Magenta
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SKILL_READY = (255, 255, 255)
        self.COLOR_SKILL_COOLDOWN = (80, 80, 80)
        self.COLOR_SKILL_LOCKED = (40, 40, 40)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.clock = pygame.time.Clock()
        
        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.player_rhythm_direction = 0 # 0:none, 1:U, 2:D, 3:L, 4:R
        self.enemies = []
        self.particles = []
        self.pulses = []
        self.active_rhythm_indicators = {}
        
        # Skill: Shield
        self.shield_unlocked = False
        self.shield_unlock_score = 100
        self.shield_active = False
        self.shield_duration = 0
        self.shield_max_duration = 5 # beats
        self.shield_cooldown = 0
        self.shield_max_cooldown = 20 # beats

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.MAX_HEALTH
        self.player_rhythm_direction = 0
        
        self.enemies = []
        self.particles = []
        self.pulses = []
        self.active_rhythm_indicators = {}
        
        self.shield_unlocked = False
        self.shield_active = False
        self.shield_duration = 0
        self.shield_cooldown = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement in [1, 2, 3, 4]:
            self.player_rhythm_direction = movement
            self.active_rhythm_indicators[movement] = 10 # visual effect duration

        # --- Update Game Logic ---
        self._update_skills(shift_held)
        self._update_pulses(space_held)
        self._spawn_enemies()
        self._update_enemies()
        
        reward += self._handle_interactions()
        self._update_player_state()
        reward += self._check_skill_unlocks()
        
        self._update_effects()
        
        # --- Termination Check ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.player_health <= 0:
            terminated = True

        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100 # Loss penalty
            else: # Survived
                reward = 100 # Win bonus
        
        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Game Logic Helpers ---
    def _update_skills(self, shift_held):
        if self.shield_cooldown > 0:
            self.shield_cooldown -= 1
        
        if self.shield_active:
            self.shield_duration -= 1
            if self.shield_duration <= 0:
                self.shield_active = False

        if shift_held and self.shield_unlocked and self.shield_cooldown == 0 and not self.shield_active:
            # SFX: Shield Activate
            self.shield_active = True
            self.shield_duration = self.shield_max_duration
            self.shield_cooldown = self.shield_max_cooldown
            for _ in range(30):
                self.particles.append(self._create_particle(
                    pos=pygame.Vector2(self.CENTER_X, self.CENTER_Y),
                    color=self.COLOR_DEFENSE,
                    lifespan=20,
                    size_range=(2, 5),
                    speed_range=(2, 6),
                    shape='ring'
                ))
    
    def _update_pulses(self, space_held):
        if space_held:
            # SFX: Pulse Charge
            self.pulses.append({'radius': 10, 'lifespan': 15, 'max_radius': self.WIDTH // 2})
    
    def _spawn_enemies(self):
        # Difficulty scaling
        glitch_chance = 0.1 + (self.steps / self.MAX_STEPS) * 0.4 # up to 50%
        virus_chance = 0.0 + (self.steps / 200) * 0.005 * 10 # up to 2.5% after 200 steps
        
        if self.np_random.random() < glitch_chance:
            self._create_enemy('glitch')
        if self.steps > 200 and self.np_random.random() < virus_chance:
            self._create_enemy('virus')

    def _create_enemy(self, type):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
        elif edge == 2: # Left
            pos = pygame.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))
        
        direction = self.np_random.integers(1, 5) # 1-4
        speed = 1.0 + (self.steps / self.MAX_STEPS) * 2.0
        if type == 'virus':
            speed *= 1.5
        
        self.enemies.append({
            'pos': pos,
            'type': type,
            'direction': direction,
            'speed': speed,
            'size': 10 if type == 'glitch' else 15
        })

    def _update_enemies(self):
        for enemy in self.enemies:
            target = pygame.Vector2(self.CENTER_X, self.CENTER_Y)
            move_dir = (target - enemy['pos']).normalize()
            enemy['pos'] += move_dir * enemy['speed']

    def _handle_interactions(self):
        reward = 0
        enemies_to_remove = []
        
        # Pulse-Enemy collisions
        for pulse in self.pulses:
            for enemy in self.enemies:
                if enemy in enemies_to_remove: continue
                dist = enemy['pos'].distance_to(pygame.Vector2(self.CENTER_X, self.CENTER_Y))
                if abs(dist - pulse['radius']) < 15: # Hit window
                    # SFX: Enemy Destroyed
                    enemies_to_remove.append(enemy)
                    reward += 5 if enemy['type'] == 'glitch' else 10
                    self.score += 5 if enemy['type'] == 'glitch' else 10
                    color = self.COLOR_GLITCH if enemy['type'] == 'glitch' else self.COLOR_VIRUS
                    for _ in range(20):
                        self.particles.append(self._create_particle(
                            pos=enemy['pos'], color=color, lifespan=25, size_range=(1, 4), speed_range=(1, 5)
                        ))

        # Rhythm defense and damage
        damage_this_step = 0
        perfect_beat = True
        
        danger_zone_radius = 80
        for enemy in self.enemies:
            if enemy in enemies_to_remove: continue
            dist = enemy['pos'].distance_to(pygame.Vector2(self.CENTER_X, self.CENTER_Y))
            
            if dist < danger_zone_radius:
                perfect_beat = False
                if self.player_rhythm_direction == enemy['direction']:
                    # SFX: Beat Match
                    reward += 1
                    # Visual feedback for parry
                    self.particles.append(self._create_particle(
                        pos=enemy['pos'], color=self.COLOR_DEFENSE, lifespan=10, size_range=(3, 6), speed_range=(0.5, 1)
                    ))
                else:
                    # SFX: Damage Taken
                    reward -= 1
                    damage = 5 if enemy['type'] == 'glitch' else 15
                    if not self.shield_active:
                        damage_this_step += damage
                    else:
                        # SFX: Shield Block
                        self.particles.append(self._create_particle(
                            pos=pygame.Vector2(self.CENTER_X, self.CENTER_Y), color=self.COLOR_DEFENSE, lifespan=15, size_range=(2,4), speed_range=(3,7), shape='ring'
                        ))
                
                # Enemy is consumed on interaction
                enemies_to_remove.append(enemy)

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        
        if damage_this_step > 0:
            self.player_health -= damage_this_step
            self.score = max(0, self.score - 1) # Small score penalty
            for _ in range(int(damage_this_step * 2)):
                self.particles.append(self._create_particle(
                    pos=pygame.Vector2(self.CENTER_X, self.CENTER_Y), color=self.COLOR_DAMAGE_ARC, lifespan=30, size_range=(2, 5), speed_range=(3, 8)
                ))
        
        # Health regen on perfect beat (no enemies in danger zone)
        if perfect_beat and self.player_health < self.MAX_HEALTH:
            self.player_health += 0.5

        return reward

    def _update_player_state(self):
        self.player_health = max(0, min(self.MAX_HEALTH, self.player_health))
        assert 0 <= self.player_health <= self.MAX_HEALTH

    def _check_skill_unlocks(self):
        reward = 0
        if not self.shield_unlocked and self.score >= self.shield_unlock_score:
            self.shield_unlocked = True
            reward += 2
            # SFX: Skill Unlock
            for _ in range(50):
                self.particles.append(self._create_particle(
                    pos=pygame.Vector2(self.WIDTH-100, self.HEIGHT-30), color=self.COLOR_SKILL_READY, lifespan=40, size_range=(1,4), speed_range=(2,6)
                ))
        return reward

    def _update_effects(self):
        # Particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] -= p['decay']

        # Pulses
        self.pulses = [p for p in self.pulses if p['lifespan'] > 0]
        for p in self.pulses:
            p['lifespan'] -= 1
            p['radius'] += (p['max_radius'] - p['radius']) * 0.2

        # Rhythm indicators
        for key in list(self.active_rhythm_indicators.keys()):
            self.active_rhythm_indicators[key] -= 1
            if self.active_rhythm_indicators[key] <= 0:
                del self.active_rhythm_indicators[key]
                
    def _create_particle(self, pos, color, lifespan, size_range, speed_range, shape='circle'):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(*speed_range)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        size = self.np_random.uniform(*size_range)
        return {
            'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 
            'max_life': lifespan, 'color': color, 'size': size, 
            'decay': size / lifespan, 'shape': shape
        }
        
    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_enemies()
        self._render_effects()
        self._render_heart()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_frame(self):
        if self.window is None: return
        obs = self._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        self.window.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_background(self):
        # Pulsing border for beat
        beat_pulse = (math.sin(self.steps * math.pi * 0.5) + 1) / 2 # Slower pulse
        pulse_color = tuple([min(255, int(c * (0.5 + beat_pulse * 0.5))) for c in self.COLOR_GRID])
        pygame.draw.rect(self.screen, pulse_color, (0, 0, self.WIDTH, self.HEIGHT), 4)

        # Grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
            
    def _render_heart(self):
        # Health-based size
        base_radius = 40
        health_ratio = self.player_health / self.MAX_HEALTH
        current_radius = int(base_radius * (0.6 + health_ratio * 0.4))
        
        # Pulsing effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        
        # Outer glow
        glow_radius = int(current_radius * 1.5 + pulse * 5)
        glow_alpha = int(80 + pulse * 40)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, glow_radius, (*self.COLOR_HEART, glow_alpha))
        
        # Main heart circle
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, current_radius, self.COLOR_HEART)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, current_radius, (*self.COLOR_HEART, 100))
        
        # Health Arc
        arc_radius = current_radius + 15
        health_angle = int(360 * health_ratio)
        if health_angle > 0:
            pygame.gfxdraw.arc(self.screen, self.CENTER_X, self.CENTER_Y, arc_radius, -90, -90 + health_angle, self.COLOR_HEALTH_ARC)
            pygame.gfxdraw.arc(self.screen, self.CENTER_X, self.CENTER_Y, arc_radius+1, -90, -90 + health_angle, self.COLOR_HEALTH_ARC)
        if health_angle < 360:
             pygame.gfxdraw.arc(self.screen, self.CENTER_X, self.CENTER_Y, arc_radius, -90 + health_angle, 270, self.COLOR_DAMAGE_ARC)
             pygame.gfxdraw.arc(self.screen, self.CENTER_X, self.CENTER_Y, arc_radius+1, -90 + health_angle, 270, self.COLOR_DAMAGE_ARC)

        # Rhythm direction indicators
        indicator_dist = current_radius + 35
        directions = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        for d, (dx, dy) in directions.items():
            pos = (self.CENTER_X + int(dx * indicator_dist), self.CENTER_Y + int(dy * indicator_dist))
            is_active = self.player_rhythm_direction == d
            is_fading = d in self.active_rhythm_indicators
            
            color = self.COLOR_DEFENSE if is_active else self.COLOR_GRID
            if is_fading and not is_active:
                fade_alpha = int(255 * (self.active_rhythm_indicators[d] / 10))
                color = (*self.COLOR_DEFENSE[:3], fade_alpha)
            
            if d in [1,2]: # Up/Down triangles
                points = [(pos[0], pos[1] - dy*5), (pos[0]-5, pos[1]), (pos[0]+5, pos[1])]
            else: # Left/Right triangles
                points = [(pos[0] - dx*5, pos[1]), (pos[0], pos[1]-5), (pos[0], pos[1]+5)]
            
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            if is_active:
                 pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy['pos'].x), int(enemy['pos'].y)
            size = int(enemy['size'])
            color = self.COLOR_GLITCH if enemy['type'] == 'glitch' else self.COLOR_VIRUS
            
            # Pulsing glow
            pulse = (math.sin(pygame.time.get_ticks() * 0.01 + x) + 1) / 2
            glow_size = int(size * 1.8 + pulse * 3)
            glow_alpha = int(100 + pulse * 50)
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_size, (*color, glow_alpha))

            # Main body
            if enemy['type'] == 'glitch':
                pygame.gfxdraw.box(self.screen, (x - size//2, y-size//2, size, size), color)
            else: # Virus
                pygame.gfxdraw.filled_circle(self.screen, x, y, size//2, color)
                pygame.gfxdraw.aacircle(self.screen, x, y, size//2, color)
            
            # Required direction indicator
            directions = {1: '▲', 2: '▼', 3: '◀', 4: '▶'}
            text = self.font_small.render(directions[enemy['direction']], True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=(x, y)))
            
    def _render_effects(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = max(0, int(p['size']))
            if p['shape'] == 'ring':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
        
        # Offensive Pulses
        for pulse in self.pulses:
            alpha = int(255 * (pulse['lifespan'] / 15))
            color = (*self.COLOR_OFFENSE, alpha)
            radius = int(pulse['radius'])
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, radius, color)
                pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, radius-1, color)
        
        # Shield
        if self.shield_active:
            pulse = (math.sin(pygame.time.get_ticks() * 0.02) + 1) / 2
            radius = 65 + int(pulse * 5)
            alpha = 100 + int(self.shield_duration / self.shield_max_duration * 155)
            color = (*self.COLOR_DEFENSE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, radius, (*color[:3], alpha//4))
            pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, radius, color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Beat
        beat_text = self.font_small.render(f"BEAT: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(beat_text, (self.WIDTH - beat_text.get_width() - 10, 10))
        
        # Skill UI
        skill_pos = (self.WIDTH - 150, self.HEIGHT - 40)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (*skill_pos, 140, 30), border_radius=5)
        
        skill_text_str = "SHIELD"
        skill_color = self.COLOR_SKILL_LOCKED
        if self.shield_unlocked:
            skill_color = self.COLOR_SKILL_COOLDOWN if self.shield_cooldown > 0 else self.COLOR_SKILL_READY
        
        skill_text = self.font_small.render(skill_text_str, True, skill_color)
        self.screen.blit(skill_text, skill_text.get_rect(center=(skill_pos[0]+70, skill_pos[1]+15)))

        if self.shield_unlocked and self.shield_cooldown > 0:
            cooldown_ratio = self.shield_cooldown / self.shield_max_cooldown
            bar_width = int(136 * cooldown_ratio)
            pygame.draw.rect(self.screen, self.COLOR_SKILL_COOLDOWN, (skill_pos[0]+2, skill_pos[1]+2, bar_width, 26), border_radius=4)
        
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "SYSTEM STABLE" if self.steps >= self.MAX_STEPS else "SYSTEM OVERLOAD"
            color = self.COLOR_HEALTH_ARC if self.steps >= self.MAX_STEPS else self.COLOR_DAMAGE_ARC
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.CENTER_X, self.CENTER_Y - 20)))
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, final_score_text.get_rect(center=(self.CENTER_X, self.CENTER_Y + 30)))

    # --- Gymnasium Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "shield_unlocked": self.shield_unlocked,
            "shield_cooldown": self.shield_cooldown
        }
        
    def close(self):
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # --- Keyboard Control Mapping ---
    # Movement: Arrow Keys
    # Space: Space Bar
    # Shift: Left Shift
    
    while not done:
        # Default action is no-op
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(3000) # Show final screen for 3 seconds
            obs, info = env.reset()

    env.close()