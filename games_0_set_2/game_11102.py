import gymnasium as gym
import os
import pygame
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:36:04.930856
# Source Brief: brief_01102.md
# Brief Index: 1102
# """import gymnasium as gym
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a "Missile Command" style game.
    The player must defend cities from incoming missile waves by building
    and managing defense batteries.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your cities from waves of incoming missiles by building and managing automated defense batteries."
    )
    user_guide = (
        "Press SPACE to build a defense battery at the highlighted location. Press SHIFT to repair the most damaged city."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_LEVEL = 350
    MAX_STEPS = 2000 # Increased from 1000 to allow for longer games
    WIN_WAVE = 20

    # Colors
    COLOR_BG = (10, 10, 26) # Dark blue
    COLOR_GROUND = (40, 30, 30)
    COLOR_CITY = (76, 175, 80) # Green
    COLOR_CITY_DAMAGED = (244, 67, 54) # Red
    COLOR_DEFENSE = (0, 229, 255) # Cyan
    COLOR_MISSILE = (255, 23, 68) # Bright Red
    COLOR_PROJECTILE = (255, 255, 255) # White
    COLOR_TEXT = (224, 224, 224)
    COLOR_RESOURCE = (255, 193, 7) # Yellow
    COLOR_BUILD_INDICATOR = (0, 229, 255, 100) # Transparent Cyan

    # Game Parameters
    INITIAL_RESOURCES = 30
    RESOURCES_PER_WAVE = 25
    COST_DEFENSE = 10
    COST_PER_HEALTH_REPAIR = 0.2 # 5 resources for 25 health
    DEFENSE_RANGE = 150
    DEFENSE_COOLDOWN = 15 # steps
    PROJECTILE_SPEED = 8.0
    EXPLOSION_RADIUS = 25
    EXPLOSION_DURATION = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode

        # Define city and defense positions
        self.city_positions = [(100, self.GROUND_LEVEL), (320, self.GROUND_LEVEL), (540, self.GROUND_LEVEL)]
        self.defense_build_spots = [(210, self.GROUND_LEVEL), (430, self.GROUND_LEVEL)]

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.resources = 0
        self.cities = []
        self.defenses = []
        self.missiles = []
        self.projectiles = []
        self.explosions = []
        self.particles = []
        self.current_build_spot_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.resources = self.INITIAL_RESOURCES
        
        self.cities = [{'pos': pos, 'health': 100.0} for pos in self.city_positions]
        self.defenses = []
        self.missiles = []
        self.projectiles = []
        self.explosions = []
        self.particles = []
        
        self.current_build_spot_index = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.screen_shake = 0

        self._start_new_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing and return final state
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0, True, False, info

        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # --- Handle Player Actions (Edge-triggered) ---
        if space_held and not self.last_space_held:
            reward += self._action_build_defense()
        if shift_held and not self.last_shift_held:
            reward += self._action_repair_city()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game Logic ---
        self._update_defenses()
        reward += self._update_projectiles()
        reward += self._update_missiles()
        self._update_effects()

        # --- Check Wave Completion ---
        if not self.missiles and not self.game_over:
            self.score += 1
            reward += 1
            if self.wave_number >= self.WIN_WAVE:
                self.game_over = True
                self.score += 100
                reward += 100
            else:
                self._start_new_wave()
        
        # --- Check Termination Conditions ---
        terminated = self.game_over
        
        # Loss condition
        if sum(c['health'] for c in self.cities) <= 0 and not terminated:
            self.game_over = True
            terminated = True
            self.score -= 100
            reward -= 100

        # Max steps condition
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _start_new_wave(self):
        self.wave_number += 1
        self.resources += self.RESOURCES_PER_WAVE
        
        num_missiles = 2 + self.wave_number
        missile_speed = 1.0 + (self.wave_number // 5) * 0.2

        for _ in range(num_missiles):
            start_x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
            start_pos = (start_x, 0)
            
            # Target can be a city or the ground between cities
            target_city = self.np_random.choice(self.cities) if self.cities else None
            if target_city and self.np_random.random() > 0.25:
                 target_pos = (target_city['pos'][0] + self.np_random.uniform(-20, 20), self.GROUND_LEVEL)
            else:
                 target_pos = (self.np_random.uniform(50, self.SCREEN_WIDTH-50), self.GROUND_LEVEL)

            self.missiles.append({
                'start_pos': start_pos,
                'pos': list(start_pos),
                'target_pos': target_pos,
                'speed': missile_speed
            })

    def _action_build_defense(self):
        if self.resources >= self.COST_DEFENSE and len(self.defenses) < len(self.defense_build_spots):
            # Check if a defense already exists at the target spot
            build_pos = self.defense_build_spots[self.current_build_spot_index]
            if any(d['pos'] == build_pos for d in self.defenses):
                self.current_build_spot_index = (self.current_build_spot_index + 1) % len(self.defense_build_spots)
                build_pos = self.defense_build_spots[self.current_build_spot_index]

            if not any(d['pos'] == build_pos for d in self.defenses):
                self.resources -= self.COST_DEFENSE
                self.defenses.append({
                    'pos': build_pos,
                    'cooldown': 0,
                    'range': self.DEFENSE_RANGE
                })
                # sfx: build_complete.wav
                self.current_build_spot_index = (self.current_build_spot_index + 1) % len(self.defense_build_spots)
                return 0.5 # Small reward for building
        # sfx: action_failed.wav
        return 0

    def _action_repair_city(self):
        cities_with_damage = [c for c in self.cities if c['health'] < 100]
        if not cities_with_damage:
            return 0

        # Find city with lowest health
        city_to_repair = min(cities_with_damage, key=lambda c: c['health'])
        health_to_restore = 100.0 - city_to_repair['health']
        cost = health_to_restore * self.COST_PER_HEALTH_REPAIR

        if self.resources >= cost:
            self.resources -= cost
            city_to_repair['health'] = 100.0
            # sfx: repair.wav
            return 0.2 # Small reward for maintaining cities
        # sfx: action_failed.wav
        return 0

    def _update_defenses(self):
        for defense in self.defenses:
            defense['cooldown'] = max(0, defense['cooldown'] - 1)
            if defense['cooldown'] == 0 and self.missiles:
                # Find closest missile in range
                targets_in_range = []
                for missile in self.missiles:
                    dist = math.hypot(missile['pos'][0] - defense['pos'][0], missile['pos'][1] - defense['pos'][1])
                    if dist <= defense['range']:
                        targets_in_range.append((dist, missile))
                
                if targets_in_range:
                    closest_missile = min(targets_in_range, key=lambda t: t[0])[1]
                    self.projectiles.append({
                        'pos': list(defense['pos']),
                        'target': closest_missile,
                        'speed': self.PROJECTILE_SPEED
                    })
                    defense['cooldown'] = self.DEFENSE_COOLDOWN
                    # sfx: fire_projectile.wav

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p['target'] not in self.missiles:
                # Target is already destroyed, fizzle out
                self.projectiles.remove(p)
                continue
            
            target_pos = p['target']['pos']
            angle = math.atan2(target_pos[1] - p['pos'][1], target_pos[0] - p['pos'][0])
            p['pos'][0] += math.cos(angle) * p['speed']
            p['pos'][1] += math.sin(angle) * p['speed']

            if math.hypot(p['pos'][0] - target_pos[0], p['pos'][1] - target_pos[1]) < 10:
                self._create_explosion(target_pos, self.EXPLOSION_RADIUS, self.COLOR_MISSILE)
                # sfx: missile_intercepted.wav
                self.missiles.remove(p['target'])
                self.projectiles.remove(p)
                self.score += 0.1
                reward += 0.1
        return reward

    def _update_missiles(self):
        reward = 0
        for m in self.missiles[:]:
            angle = math.atan2(m['target_pos'][1] - m['start_pos'][1], m['target_pos'][0] - m['start_pos'][0])
            m['pos'][0] += math.cos(angle) * m['speed']
            m['pos'][1] += math.sin(angle) * m['speed']
            
            if m['pos'][1] >= self.GROUND_LEVEL:
                self.missiles.remove(m)
                self._create_explosion(m['pos'], self.EXPLOSION_RADIUS * 1.5, self.COLOR_CITY_DAMAGED)
                self.screen_shake = 10
                # sfx: city_impact.wav
                
                # Check for city damage
                damage = 25
                hit_city = False
                for city in self.cities:
                    if abs(city['pos'][0] - m['pos'][0]) < 40 and city['health'] > 0:
                        damage_dealt = min(city['health'], damage)
                        city['health'] -= damage_dealt
                        reward -= damage_dealt * 0.1
                        self.score -= damage_dealt * 0.1
                        hit_city = True
                if not hit_city:
                    reward -= 0.1 # Penalty for ground hit near cities
        return reward

    def _update_effects(self):
        # Update explosions
        for exp in self.explosions[:]:
            exp['life'] -= 1
            if exp['life'] <= 0:
                self.explosions.remove(exp)
        
        # Update particles
        for part in self.particles[:]:
            part['pos'][0] += part['vel'][0]
            part['pos'][1] += part['vel'][1]
            part['life'] -= 1
            if part['life'] <= 0:
                self.particles.remove(part)
        
        # Update screen shake
        self.screen_shake = max(0, self.screen_shake - 1)

    def _create_explosion(self, pos, radius, color):
        self.explosions.append({'pos': pos, 'radius': radius, 'life': self.EXPLOSION_DURATION, 'color': color})
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': random.choice([self.COLOR_RESOURCE, color, (200,200,200)])
            })

    def _get_observation(self):
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake)
            offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake)
        
        # Create a temporary surface to draw on with the offset
        render_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # Background
        render_surface.fill(self.COLOR_BG)
        pygame.draw.rect(render_surface, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_LEVEL))

        self._render_game(render_surface)
        
        # Apply screen shake by blitting with an offset
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(render_surface, (offset_x, offset_y))

        self._render_ui(self.screen) # UI is not affected by shake
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        # Draw cities
        for city in self.cities:
            if city['health'] > 0:
                health_color = self.COLOR_CITY
                if city['health'] < 100:
                    health_color = pygame.Color(self.COLOR_CITY).lerp(self.COLOR_CITY_DAMAGED, 1 - city['health'] / 100.0)
                pygame.draw.rect(surface, health_color, (city['pos'][0] - 20, city['pos'][1] - 15, 40, 15))
                pygame.draw.rect(surface, tuple(c/2 for c in health_color), (city['pos'][0] - 20, city['pos'][1] - 15, 40, 15), 2)


        # Draw defenses
        for defense in self.defenses:
            pos = defense['pos']
            points = [(pos[0], pos[1] - 20), (pos[0] - 15, pos[1]), (pos[0] + 15, pos[1])]
            pygame.gfxdraw.aapolygon(surface, points, self.COLOR_DEFENSE)
            pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_DEFENSE)
            # Cooldown indicator
            cooldown_ratio = defense['cooldown'] / self.DEFENSE_COOLDOWN
            if cooldown_ratio > 0:
                pygame.draw.arc(surface, (255,0,0), (pos[0]-10, pos[1]-10, 20, 20), 0, cooldown_ratio * 2 * math.pi, 3)

        # Draw next build spot indicator
        if len(self.defenses) < len(self.defense_build_spots):
            build_pos = self.defense_build_spots[self.current_build_spot_index]
            if not any(d['pos'] == build_pos for d in self.defenses):
                points = [(build_pos[0], build_pos[1] - 20), (build_pos[0] - 15, build_pos[1]), (build_pos[0] + 15, build_pos[1])]
                pygame.gfxdraw.aapolygon(surface, points, self.COLOR_BUILD_INDICATOR)


        # Draw missiles
        for m in self.missiles:
            start_pos_int = (int(m['start_pos'][0]), int(m['start_pos'][1]))
            end_pos_int = (int(m['pos'][0]), int(m['pos'][1]))
            pygame.draw.line(surface, (100,0,0), start_pos_int, end_pos_int, 1) # Faint trail
            pygame.draw.aaline(surface, self.COLOR_MISSILE, start_pos_int, end_pos_int, 2)
            pygame.gfxdraw.filled_circle(surface, end_pos_int[0], end_pos_int[1], 3, self.COLOR_PROJECTILE)

        # Draw projectiles
        for p in self.projectiles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 3, self.COLOR_PROJECTILE)
            for i in range(1, 5):
                alpha = 255 - i * 50
                pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], 3 + i, (*self.COLOR_DEFENSE, alpha))

        # Draw particles and explosions
        for part in self.particles:
            pygame.draw.circle(surface, part['color'], (int(part['pos'][0]), int(part['pos'][1])), int(part['life']/10 + 1))
        for exp in self.explosions:
            progress = (self.EXPLOSION_DURATION - exp['life']) / self.EXPLOSION_DURATION
            current_radius = int(exp['radius'] * math.sin(progress * math.pi))
            alpha = int(255 * (1 - progress))
            if current_radius > 0:
                # Draw a filled circle with alpha
                temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*exp['color'], alpha), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(exp['pos'][0]) - current_radius, int(exp['pos'][1]) - current_radius))


    def _render_ui(self, surface):
        # Resources
        res_text = self.font_small.render(f"RES: {int(self.resources)}", True, self.COLOR_RESOURCE)
        surface.blit(res_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        surface.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # City Health
        for i, city in enumerate(self.cities):
            health_text = self.font_small.render(f"{int(city['health'])}%", True, self.COLOR_TEXT)
            text_pos = (city['pos'][0] - health_text.get_width() // 2, self.GROUND_LEVEL + 5)
            surface.blit(health_text, text_pos)
        
        # Game Over / Win Message
        if self.game_over:
            if sum(c['health'] for c in self.cities) <= 0:
                msg = "GAME OVER"
                color = self.COLOR_CITY_DAMAGED
            else:
                msg = "YOU WIN!"
                color = self.COLOR_DEFENSE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            surface.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "cities_health": [c['health'] for c in self.cities],
            "num_defenses": len(self.defenses)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Temporarily set up state for one-time observation generation
        self.wave_number = 1
        self.resources = self.INITIAL_RESOURCES
        self.cities = [{'pos': pos, 'health': 100.0} for pos in self.city_positions]
        self._start_new_wave()
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Missile Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action defaults
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()