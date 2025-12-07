import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:42:14.450883
# Source Brief: brief_03445.md
# Brief Index: 3445
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player assembles asteroid fragments into stable spaceships.
    The core challenge lies in managing the physics-based stability of the growing ship
    by carefully placing fragments and tilting the construction platform.

    Visuals:
    - Clean, futuristic aesthetic with a dark space background.
    - Fragments and ships are rendered as anti-aliased polygons with glow effects.
    - A dynamic UI provides feedback on stability, score, and current objective.
    - Particle effects provide satisfying feedback for explosions.

    Gameplay:
    - Select and move fragments using directional actions.
    - Place fragments onto the ship structure using the 'space' action.
    - Counteract instability by tilting the construction platform with the 'shift' action.
    - Successfully completing ship blueprints unlocks more complex designs.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Assemble spaceships by placing asteroid fragments onto a construction platform. "
        "Carefully manage the ship's stability by balancing mass and tilting the platform to counteract wobble."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the selected fragment. "
        "Press space to place the fragment and shift to tilt the construction platform."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (200, 200, 220)
    COLOR_PLATFORM = (100, 110, 140)
    COLOR_GHOST = (60, 80, 120, 100)
    COLOR_SELECTED = (0, 150, 255)
    COLOR_STABLE = (0, 255, 150)
    COLOR_UNSTABLE = (255, 255, 0)
    COLOR_CRITICAL = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)

    # Game Parameters
    MAX_EPISODE_STEPS = 2000
    PLAYER_SPEED = 8.0
    PLATFORM_Y = 350
    PLATFORM_TILT_INCREMENT = math.radians(2.0)
    MAX_PLATFORM_TILT = math.radians(40.0)
    INSTABILITY_THRESHOLD = 1.0
    STABLE_THRESHOLD = 0.4 # Must be below this to be considered a success

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)

        self._define_ship_designs()
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
            for _ in range(150)
        ]

        # Persistent state (carries over episodes)
        self.persistent_score = 0
        self.unlocked_level = 0
        self.successful_assemblies = 0

        # Initialize episode-specific state variables to avoid attribute errors
        self.steps = 0
        self.game_over = False
        self.outcome = ""
        self.platform_tilt = 0.0
        self.tilt_direction = 1
        self.last_space_held = False
        self.last_shift_held = False
        self.placed_fragments = []
        self.selected_fragment = None
        self.fragments_to_place = []
        self.ship_center_of_mass = pygame.Vector2(self.WIDTH / 2, self.PLATFORM_Y)
        self.ship_instability = 0.0
        self.ship_wobble_angle = 0.0
        self.ship_wobble_speed = 0.0
        self.particles = []
        self.reward_this_step = 0

    def _define_ship_designs(self):
        self.SHIP_DESIGNS = []
        # Level 0: Simple 'I' shape
        self.SHIP_DESIGNS.append([
            {'offset': (0, 0), 'mass': 2},
            {'offset': (0, -40), 'mass': 1},
        ])
        # Level 1: 'L' shape
        self.SHIP_DESIGNS.append([
            {'offset': (0, 0), 'mass': 2},
            {'offset': (0, -40), 'mass': 1},
            {'offset': (40, 0), 'mass': 1.5},
        ])
        # Level 2: 'T' shape
        self.SHIP_DESIGNS.append([
            {'offset': (0, 0), 'mass': 2},
            {'offset': (0, -40), 'mass': 1},
            {'offset': (40, -40), 'mass': 1},
            {'offset': (-40, -40), 'mass': 1},
        ])
        # Add more complex designs
        for i in range(3, 10):
            num_frags = 4 + i
            design = [{'offset': (0, 0), 'mass': 2}]
            for j in range(num_frags -1):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(30, 80)
                design.append({
                    'offset': (math.cos(angle) * radius, math.sin(angle) * radius),
                    'mass': random.uniform(0.8, 1.8)
                })
            self.SHIP_DESIGNS.append(design)

    def _generate_fragment_shape(self, radius, points=7):
        shape = []
        for i in range(points):
            angle = (i / points) * 2 * math.pi
            r = radius * random.uniform(0.7, 1.1)
            shape.append((math.cos(angle) * r, math.sin(angle) * r))
        return shape

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.game_over = False
        self.outcome = ""
        self.platform_tilt = 0.0
        self.tilt_direction = 1
        self.last_space_held = False
        self.last_shift_held = False

        self.placed_fragments = []
        self.ship_center_of_mass = pygame.Vector2(self.WIDTH / 2, self.PLATFORM_Y - 20)
        self.ship_instability = 0.0
        self.ship_wobble_angle = 0.0
        self.ship_wobble_speed = 0.0
        self.particles = []

        self._load_current_level()

        return self._get_observation(), self._get_info()

    def _load_current_level(self):
        level_idx = min(self.unlocked_level, len(self.SHIP_DESIGNS) - 1)
        design = self.SHIP_DESIGNS[level_idx]
        self.fragments_to_place = []
        for frag_info in design:
            frag = {
                'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
                'shape': self._generate_fragment_shape(20),
                'mass': frag_info['mass'],
                'target_offset': pygame.Vector2(frag_info['offset'])
            }
            self.fragments_to_place.append(frag)

        self.selected_fragment = self.fragments_to_place.pop(0)

    def step(self, action):
        if self.game_over:
            # If game is over, an action should do nothing but allow for reset
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self._handle_actions(action)
        self._update_physics()
        self._update_particles()
        
        self.steps += 1
        
        # Continuous reward/penalty
        self.reward_this_step -= 0.01 * abs(math.degrees(self.platform_tilt))

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            self.game_over = True
            self.outcome = "TIME OUT"
            self.reward_this_step += -1 # Small penalty for timeout
            
        terminated = terminated or truncated

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False, # Using terminated instead of truncated for gym API
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # 1. Movement
        if self.selected_fragment:
            if movement == 1: self.selected_fragment['pos'].y -= self.PLAYER_SPEED
            elif movement == 2: self.selected_fragment['pos'].y += self.PLAYER_SPEED
            elif movement == 3: self.selected_fragment['pos'].x -= self.PLAYER_SPEED
            elif movement == 4: self.selected_fragment['pos'].x += self.PLAYER_SPEED
            
            # Clamp to screen
            self.selected_fragment['pos'].x = max(20, min(self.WIDTH - 20, self.selected_fragment['pos'].x))
            self.selected_fragment['pos'].y = max(20, min(self.HEIGHT - 20, self.selected_fragment['pos'].y))

        # 2. Place fragment
        if space_pressed and self.selected_fragment:
            # Place fragment relative to the current ship's center of mass
            self.selected_fragment['pos'] = self.ship_center_of_mass + self.selected_fragment['target_offset']
            self.placed_fragments.append(self.selected_fragment)
            # sound: fragment_attach.wav
            self.reward_this_step += 0.1

            if self.fragments_to_place:
                self.selected_fragment = self.fragments_to_place.pop(0)
            else:
                self.selected_fragment = None
        
        # 3. Tilt platform
        if shift_pressed:
            self.platform_tilt += self.PLATFORM_TILT_INCREMENT * self.tilt_direction
            self.platform_tilt = max(-self.MAX_PLATFORM_TILT, min(self.MAX_PLATFORM_TILT, self.platform_tilt))
            self.tilt_direction *= -1
            # sound: platform_tilt.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_physics(self):
        if not self.placed_fragments:
            return

        # 1. Calculate new center of mass (CoM)
        total_mass = 0
        com_x, com_y = 0, 0
        for frag in self.placed_fragments:
            total_mass += frag['mass']
            com_x += frag['pos'].x * frag['mass']
            com_y += frag['pos'].y * frag['mass']
        
        if total_mass > 0:
            self.ship_center_of_mass = pygame.Vector2(com_x / total_mass, com_y / total_mass)

        # 2. Calculate instability based on CoM distance from platform center and tilt
        platform_center_x = self.WIDTH / 2
        com_offset = self.ship_center_of_mass.x - platform_center_x
        
        # Torque-like force due to tilt and offset CoM
        instability_force = com_offset * math.sin(self.platform_tilt) * 0.001
        
        # Add to wobble speed (angular acceleration)
        self.ship_wobble_speed += instability_force
        
        # Apply damping
        self.ship_wobble_speed *= 0.95
        
        # Update wobble angle (angular velocity)
        self.ship_wobble_angle += self.ship_wobble_speed
        
        # Add restoring force (spring-like)
        self.ship_wobble_angle *= 0.98
        
        # 3. Update overall instability metric
        self.ship_instability = (abs(self.ship_wobble_angle) * 0.2 + abs(self.ship_wobble_speed) * 10)
        self.ship_instability = min(self.ship_instability, 1.5) # Cap for rendering

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.98

    def _check_termination(self):
        # 1. Explosion
        if self.ship_instability > self.INSTABILITY_THRESHOLD:
            self.game_over = True
            self.outcome = "EXPLOSION"
            self.reward_this_step += -10
            self._create_explosion(self.ship_center_of_mass, 100)
            # sound: explosion.wav
            return True

        # 2. Success
        if not self.selected_fragment and not self.fragments_to_place:
            # Check stability one last time
            if self.ship_instability <= self.STABLE_THRESHOLD:
                self.game_over = True
                self.outcome = "SUCCESS"
                self.reward_this_step += 10
                self.persistent_score += 1
                self.successful_assemblies += 1

                if self.successful_assemblies > 0 and self.successful_assemblies % 5 == 0:
                    self.unlocked_level += 1
                
                if abs(math.degrees(self.platform_tilt)) <= 5:
                    self.reward_this_step += 50
                # sound: success.wav
                return True
            # If all fragments are placed but it's not stable, the episode continues until it stabilizes or explodes

        # 3. Timeout is handled in step()
        
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_platform()
        self._render_target_outline()
        self._render_ship()
        self._render_selected_fragment()
        self._render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.persistent_score,
            "steps": self.steps,
            "level": self.unlocked_level,
            "instability": self.ship_instability,
        }

    def _render_background(self):
        for star_pos in self.stars:
            pygame.gfxdraw.pixel(self.screen, star_pos[0], star_pos[1], self.COLOR_STAR)

    def _render_platform(self):
        angle = self.platform_tilt
        center_x, center_y = self.WIDTH / 2, self.PLATFORM_Y
        length = self.WIDTH * 0.8
        
        start_pos = (
            center_x - length / 2 * math.cos(angle),
            center_y - length / 2 * math.sin(angle)
        )
        end_pos = (
            center_x + length / 2 * math.cos(angle),
            center_y + length / 2 * math.sin(angle)
        )
        pygame.draw.line(self.screen, self.COLOR_PLATFORM, start_pos, end_pos, 5)

    def _render_target_outline(self):
        if not self.placed_fragments and self.selected_fragment:
            base_pos = self.selected_fragment['pos']
            all_frags = [self.selected_fragment] + self.fragments_to_place
            for frag in all_frags:
                pos = base_pos + frag['target_offset']
                self._draw_polygon(pos, 0, frag['shape'], self.COLOR_GHOST, is_filled=False, width=1)

    def _render_ship(self):
        if not self.placed_fragments:
            return

        # Determine color based on instability
        if self.ship_instability < 0.4:
            color = self.COLOR_STABLE
        elif self.ship_instability < 0.75:
            color = self.COLOR_UNSTABLE
        else:
            color = self.COLOR_CRITICAL

        # Apply wobble
        wobble = self.ship_wobble_angle
        for frag in self.placed_fragments:
            # Rotate fragment position around the CoM
            rel_pos = frag['pos'] - self.ship_center_of_mass
            rotated_rel_pos = rel_pos.rotate(math.degrees(wobble))
            draw_pos = self.ship_center_of_mass + rotated_rel_pos
            self._draw_polygon(draw_pos, wobble, frag['shape'], color)

    def _render_selected_fragment(self):
        if self.selected_fragment:
            frag = self.selected_fragment
            self._draw_polygon(frag['pos'], 0, frag['shape'], self.COLOR_SELECTED, glow=True)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            if p['radius'] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
                )

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.persistent_score}", (10, 10), self.font_large)
        level_idx = min(self.unlocked_level, len(self.SHIP_DESIGNS) - 1)
        self._draw_text(f"BLUEPRINT: {level_idx + 1}", (self.WIDTH - 150, 10), self.font_large)

        # Stability Meter
        bar_width, bar_height = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_width) / 2, 15
        
        fill_ratio = min(1.0, self.ship_instability / self.INSTABILITY_THRESHOLD)
        fill_width = bar_width * fill_ratio
        
        # Interpolate color from green to red
        bar_color = (
            min(255, self.COLOR_STABLE[0] + (self.COLOR_CRITICAL[0] - self.COLOR_STABLE[0]) * fill_ratio),
            min(255, self.COLOR_STABLE[1] - (self.COLOR_STABLE[1] - self.COLOR_CRITICAL[1]) * fill_ratio),
            min(255, self.COLOR_STABLE[2] - (self.COLOR_STABLE[2] - self.COLOR_CRITICAL[2]) * fill_ratio)
        )
        
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        self._draw_text("STABILITY", (bar_x + bar_width / 2, bar_y + 30), self.font_small, center=True)

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        color = self.COLOR_CRITICAL if self.outcome == "EXPLOSION" else self.COLOR_STABLE
        self._draw_text(self.outcome, (self.WIDTH/2, self.HEIGHT/2 - 30), self.font_title, color=color, center=True)
        self._draw_text("Perform any action to reset", (self.WIDTH/2, self.HEIGHT/2 + 20), self.font_large, center=True)

    def _draw_polygon(self, center_pos, angle, points, color, is_filled=True, width=0, glow=False):
        transformed_points = []
        for p in points:
            vec = pygame.Vector2(p)
            rotated_vec = vec.rotate(math.degrees(angle))
            transformed_points.append(center_pos + rotated_vec)
        
        int_points = [(int(p.x), int(p.y)) for p in transformed_points]

        if len(int_points) < 3: return

        if glow:
            for i in range(5, 0, -1):
                glow_color = (color[0], color[1], color[2], 20)
                fat_points = []
                for p in points:
                    vec = pygame.Vector2(p) * (1 + i * 0.1)
                    rotated_vec = vec.rotate(math.degrees(angle))
                    fat_points.append(center_pos + rotated_vec)
                pygame.gfxdraw.aapolygon(self.screen, fat_points, glow_color)
        
        if is_filled:
            pygame.gfxdraw.filled_polygon(self.screen, int_points, color)
        
        outline_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
        pygame.gfxdraw.aapolygon(self.screen, int_points, outline_color if width == 0 else color)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        shadow_surface = font.render(text, True, shadow_color)
        shadow_rect = shadow_surface.get_rect()

        if center:
            text_rect.center = pos
            shadow_rect.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect.topleft = pos
            shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
        
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    def _create_explosion(self, pos, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': random.uniform(5, 15),
                'color': random.choice([self.COLOR_CRITICAL, self.COLOR_UNSTABLE, (255, 150, 0)]),
                'life': random.randint(20, 40),
                'max_life': 40
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and is not part of the Gymnasium environment
    # It will not be executed by the test suite, but is useful for debugging.
    # To run, execute `python your_file_name.py`
    
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Asteroid Assembler")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        # If the game is over, any key press can reset it.
        # This is a common pattern for simple arcade-style games.
        if env.game_over:
            if any(keys):
                obs, info = env.reset()
        else:
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()
    pygame.quit()