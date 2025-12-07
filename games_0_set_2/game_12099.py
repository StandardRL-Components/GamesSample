import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:19:23.596077
# Source Brief: brief_02099.md
# Brief Index: 2099
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Magnetic Flower Defense: A strategy/tower defense Gymnasium environment.

    The player places and charges magnetic flowers to create repulsive fields,
    defending a protected meadow zone from waves of invading insects. The game
    is presented in a visually rich 2D style with an emphasis on particle effects
    and smooth animations.

    Action Space: MultiDiscrete([5, 2, 2])
    - Action[0]: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - Action[1]: Place/Charge (0: released, 1: held)
    - Action[2]: Flip Polarity (0: released, 1: held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +5.0 for destroying an insect.
    - +20.0 for clearing a wave.
    - -20.0 for an insect reaching the protected zone (life lost).
    - Small continuous reward/penalty for insects moving away/towards the goal.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a meadow by placing and charging magnetic flowers to repel waves of invading insects."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a flower, "
        "or hold it over a flower to charge it. Press shift over a flower to flip its polarity."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Pygame Setup ===
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_wave = pygame.font.Font(None, 48)

        # === Game Constants ===
        # Colors
        self.COLOR_BG = (15, 25, 20)
        self.COLOR_GRASS = (30, 60, 40)
        self.COLOR_PROTECTED_ZONE = (40, 80, 60, 100)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_POS = (100, 255, 100)
        self.COLOR_NEG = (255, 100, 100)
        self.COLOR_NEUTRAL = (150, 150, 180)
        self.COLOR_POS_FIELD = (100, 255, 100, 0)
        self.COLOR_NEG_FIELD = (255, 100, 100, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEART = (255, 80, 80)
        # Insect Colors
        self.INSECT_COLORS = {
            'basic': (200, 150, 100),
            'pos_resist': (230, 230, 80),
            'neg_resist': (200, 100, 200)
        }
        # Mechanics
        self.MAX_STEPS = 3000
        self.MAX_LIVES = 5
        self.CURSOR_SPEED = 8.0
        self.MAX_FLOWERS = 6
        self.FLOWER_RADIUS = 12
        self.FLOWER_MAX_CHARGE = 100.0
        self.FLOWER_CHARGE_RATE = 2.0
        self.FLOWER_PLACE_COOLDOWN = 15 # frames
        self.PROTECTED_ZONE_Y = self.HEIGHT - 30
        self.BASE_INSECTS_PER_WAVE = 2
        self.BASE_INSECT_SPEED = 0.8
        self.INSECT_SPAWN_Y = -20
        self.MAGNETIC_FORCE_CONSTANT = 80.0
        self.INSECT_DESTRUCTION_DISTANCE = 20.0
        # Rewards
        self.REWARD_DESTROY_INSECT = 5.0
        self.REWARD_WAVE_CLEAR = 20.0
        self.REWARD_LIFE_LOST = 20.0
        self.REWARD_SHAPING_FACTOR = 0.02

        # === Game State ===
        self.cursor_pos = None
        self.flowers = []
        self.insects = []
        self.particles = []
        self.background_stars = []
        self.wave = 0
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.flower_place_timer = 0
        
        self._generate_background_stars()
        # self.reset() is called by the wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2 - 50], dtype=float)
        self.flowers = []
        self.insects = []
        self.particles = []
        self.wave = 0
        self.lives = self.MAX_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.flower_place_timer = 0

        self._start_new_wave()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.flower_place_timer = max(0, self.flower_place_timer - 1)
        
        reward = 0
        self._handle_input(action)
        reward += self._update_insects()
        self._update_particles()

        if not self.insects and not self.game_over:
            reward += self.REWARD_WAVE_CLEAR
            self.score += self.REWARD_WAVE_CLEAR
            self._start_new_wave()
            # sfx: wave_cleared_sound

        if self.lives <= 0:
            self.game_over = True
            # sfx: game_over_sound

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        move_vec = np.array([0, 0], dtype=float)
        if movement == 1: move_vec[1] -= 1 # Up
        elif movement == 2: move_vec[1] += 1 # Down
        elif movement == 3: move_vec[0] -= 1 # Left
        elif movement == 4: move_vec[0] += 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.cursor_pos += move_vec * self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.PROTECTED_ZONE_Y)

        # --- Button Presses (Rising Edge Detection) ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # --- Interaction Logic ---
        flower_under_cursor = self._get_flower_at(self.cursor_pos)
        
        if flower_under_cursor:
            # Charge existing flower
            if space_held:
                flower = flower_under_cursor
                if flower['charge'] < self.FLOWER_MAX_CHARGE:
                    flower['charge'] = min(self.FLOWER_MAX_CHARGE, flower['charge'] + self.FLOWER_CHARGE_RATE)
                    # sfx: charging_sound_loop
            # Flip polarity
            if shift_press:
                flower_under_cursor['polarity'] *= -1
                self._create_particles(flower_under_cursor['pos'], 15, self.COLOR_POS if flower_under_cursor['polarity'] == 1 else self.COLOR_NEG)
                # sfx: polarity_flip_sound
        else:
            # Place new flower
            if space_press and len(self.flowers) < self.MAX_FLOWERS and self.flower_place_timer == 0:
                self.flowers.append({
                    'pos': self.cursor_pos.copy(),
                    'charge': 0.0,
                    'polarity': 1, # Start as positive
                    'id': self.steps + self.np_random.random()
                })
                self.flower_place_timer = self.FLOWER_PLACE_COOLDOWN
                # sfx: place_flower_sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_insects(self):
        step_reward = 0
        insects_to_remove = []

        for insect in self.insects:
            dist_to_goal_before = self.PROTECTED_ZONE_Y - insect['pos'][1]
            
            total_force = np.array([0.0, 0.0])
            for flower in self.flowers:
                if flower['charge'] > 0:
                    vec_to_insect = insect['pos'] - flower['pos']
                    dist_sq = vec_to_insect[0]**2 + vec_to_insect[1]**2
                    if dist_sq < 1: continue # Avoid division by zero

                    # Check for resistance
                    interaction = 1.0
                    if (flower['polarity'] == 1 and insect['type'] == 'pos_resist') or \
                       (flower['polarity'] == -1 and insect['type'] == 'neg_resist'):
                        interaction = 0.0

                    if interaction > 0:
                        force_mag = interaction * self.MAGNETIC_FORCE_CONSTANT * flower['charge'] / dist_sq
                        force_dir = vec_to_insect / np.sqrt(dist_sq)
                        total_force += force_dir * force_mag

                        # Check for destruction
                        if np.sqrt(dist_sq) < self.INSECT_DESTRUCTION_DISTANCE and force_mag > 1.0:
                            if insect not in insects_to_remove:
                                step_reward += self.REWARD_DESTROY_INSECT
                                self.score += 1
                                self._create_particles(insect['pos'], 40, self.INSECT_COLORS[insect['type']])
                                insects_to_remove.append(insect)
                                # sfx: insect_zap_sound
                            continue
            
            # Apply forces and base speed
            insect['vel'] = np.array([0.0, insect['speed']]) + total_force
            insect['pos'] += insect['vel']

            # Boundary checks
            insect['pos'][0] = np.clip(insect['pos'][0], 0, self.WIDTH)
            if insect['pos'][1] < self.INSECT_SPAWN_Y: insect['pos'][1] = self.INSECT_SPAWN_Y

            # Reward shaping
            dist_to_goal_after = self.PROTECTED_ZONE_Y - insect['pos'][1]
            step_reward += (dist_to_goal_before - dist_to_goal_after) * self.REWARD_SHAPING_FACTOR

            # Check for reaching protected zone
            if insect['pos'][1] >= self.PROTECTED_ZONE_Y:
                if insect not in insects_to_remove:
                    self.lives -= 1
                    step_reward -= self.REWARD_LIFE_LOST
                    insects_to_remove.append(insect)
                    self._create_particles(insect['pos'], 20, self.COLOR_HEART)
                    # sfx: life_lost_sound

        self.insects = [i for i in self.insects if i not in insects_to_remove]
        return step_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _start_new_wave(self):
        self.wave += 1
        num_insects = self.BASE_INSECTS_PER_WAVE + self.wave - 1
        speed = self.BASE_INSECT_SPEED + 0.05 * ((self.wave - 1) // 2)

        for _ in range(num_insects):
            insect_type = 'basic'
            if self.wave >= 3:
                insect_type = self.np_random.choice(['basic', 'pos_resist'], p=[0.7, 0.3])
            if self.wave >= 6:
                insect_type = self.np_random.choice(['basic', 'pos_resist', 'neg_resist'], p=[0.5, 0.25, 0.25])
            
            self.insects.append({
                'pos': np.array([self.np_random.uniform(50, self.WIDTH - 50), float(self.INSECT_SPAWN_Y)]),
                'vel': np.array([0.0, 0.0]),
                'speed': speed,
                'type': insect_type,
                'id': self.steps + self.np_random.random()
            })

    def _get_flower_at(self, pos):
        for flower in self.flowers:
            if np.linalg.norm(pos - flower['pos']) < self.FLOWER_RADIUS:
                return flower
        return None

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })
    
    def _generate_background_stars(self):
        self.background_stars = []
        for _ in range(150):
            self.background_stars.append({
                'pos': np.array([random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)]),
                'size': random.randint(1, 2),
                'speed': random.uniform(0.1, 0.3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_magnetic_fields()
        self._render_flowers()
        self._render_insects()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Parallax stars
        for star in self.background_stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.HEIGHT:
                star['pos'][1] = 0
                star['pos'][0] = random.randint(0, self.WIDTH)
            pygame.draw.circle(self.screen, (100, 120, 110), star['pos'], star['size'])

        # Protected zone
        zone_surface = pygame.Surface((self.WIDTH, self.HEIGHT - self.PROTECTED_ZONE_Y), pygame.SRCALPHA)
        zone_surface.fill(self.COLOR_PROTECTED_ZONE)
        self.screen.blit(zone_surface, (0, self.PROTECTED_ZONE_Y))
        pygame.draw.line(self.screen, (80, 160, 120), (0, self.PROTECTED_ZONE_Y), (self.WIDTH, self.PROTECTED_ZONE_Y), 1)

        # Grass
        for i in range(100):
            x = (i * 23) % self.WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRASS, (x, self.HEIGHT), (x + random.randint(-1, 1), self.HEIGHT - random.randint(5, 15)), 1)
    
    def _render_magnetic_fields(self):
        for flower in self.flowers:
            if flower['charge'] <= 0: continue
            
            strength = flower['charge'] / self.FLOWER_MAX_CHARGE
            max_r = self.FLOWER_RADIUS + 10 + strength * 80
            color = self.COLOR_POS_FIELD if flower['polarity'] == 1 else self.COLOR_NEG_FIELD
            
            for i in range(1, 5):
                radius = int(max_r * (i / 4.0))
                alpha = int(max(0, min(255, (60 * strength) * (1 - i/5.0))))
                if alpha > 0:
                    anim_offset = math.sin(self.steps * 0.05 + i) * 0.2
                    start_angle = anim_offset
                    end_angle = math.pi + anim_offset
                    try:
                        pygame.gfxdraw.arc(self.screen, int(flower['pos'][0]), int(flower['pos'][1]), radius, int(start_angle*180/math.pi), int(end_angle*180/math.pi), (*color[:3], alpha))
                        pygame.gfxdraw.arc(self.screen, int(flower['pos'][0]), int(flower['pos'][1]), radius, int((start_angle+math.pi)*180/math.pi), int((end_angle+math.pi)*180/math.pi), (*color[:3], alpha))
                    except (ValueError, TypeError): # Catch potential errors from extreme anim_offset
                        pass

    def _render_flowers(self):
        for flower in self.flowers:
            pos = (int(flower['pos'][0]), int(flower['pos'][1]))
            charge_ratio = flower['charge'] / self.FLOWER_MAX_CHARGE
            
            # Color and glow
            base_color = self.COLOR_POS if flower['polarity'] == 1 else self.COLOR_NEG
            if charge_ratio == 0: base_color = self.COLOR_NEUTRAL
            
            glow_radius = int(self.FLOWER_RADIUS * (1 + charge_ratio * 0.5))
            glow_alpha = int(100 * charge_ratio)
            if glow_alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*base_color, glow_alpha))
            
            # Core
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FLOWER_RADIUS, base_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FLOWER_RADIUS-1, base_color)
            
            # Center dot
            center_brightness = 150 + 105 * charge_ratio
            center_color = (int(center_brightness), int(center_brightness), int(center_brightness))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, center_color)

            # Charge bar
            if charge_ratio > 0:
                bar_w, bar_h = 30, 4
                bar_x, bar_y = pos[0] - bar_w/2, pos[1] - self.FLOWER_RADIUS - 12
                fill_w = bar_w * charge_ratio
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, base_color, (bar_x, bar_y, fill_w, bar_h))

    def _render_insects(self):
        for insect in self.insects:
            pos = (int(insect['pos'][0]), int(insect['pos'][1]))
            color = self.INSECT_COLORS[insect['type']]
            
            # Body
            body_w, body_h = 10, 14
            body_rect = pygame.Rect(pos[0] - body_w/2, pos[1] - body_h/2, body_w, body_h)
            pygame.draw.ellipse(self.screen, color, body_rect)
            
            # Eyes
            eye_y = pos[1] - 2
            pygame.draw.circle(self.screen, (0,0,0), (pos[0]-3, eye_y), 2)
            pygame.draw.circle(self.screen, (0,0,0), (pos[0]+3, eye_y), 2)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / 30.0
            size = int(3 * life_ratio)
            if size > 0:
                alpha = int(255 * life_ratio)
                color = (*p['color'], alpha)
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        size = 8
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0] - size, pos[1]), (pos[0] + size, pos[1]), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (pos[0], pos[1] - size), (pos[0], pos[1] + size), 2)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_wave.render(f"WAVE {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (20, 15))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 60))

        # Lives (Hearts)
        for i in range(self.lives):
            pos = (self.WIDTH - 30 - i * 25, 25)
            pygame.gfxdraw.filled_polygon(self.screen, [(pos[0], pos[1]+5), (pos[0]-10, pos[1]-5), (pos[0]-5, pos[1]-10), (pos[0], pos[1]-5), (pos[0]+5, pos[1]-10), (pos[0]+10, pos[1]-5)], self.COLOR_HEART)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "lives": self.lives}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Magnetic Flower Defense")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    # Game loop
    while not terminated:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        terminated = terminated or truncated
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
    env.close()