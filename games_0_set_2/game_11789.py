import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:36:50.667343
# Source Brief: brief_01789.md
# Brief Index: 1789
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
        "Defend your planet by terraforming quadrants for resources and closing dimensional portals. "
        "Enter glyph sequences to repel cosmic horrors before they overwhelm your world."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a quadrant. Press space to terraform. "
        "When a portal appears, use arrow keys to select a glyph and press shift to enter it into the sequence."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Sizing
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLANET_RADIUS = 120
    PLANET_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_STAR = (150, 150, 180)
    COLOR_PLANET_OUTLINE = (80, 80, 120)
    COLOR_TERRAFORM = (50, 220, 150)
    COLOR_PORTAL = (180, 50, 255)
    COLOR_HORROR = (255, 40, 80)
    COLOR_DEFENSE_UI = (60, 180, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (40, 200, 120)
    COLOR_HEALTH_BAR_BG = (80, 20, 40)
    
    # Game Parameters
    MAX_STEPS = 5000
    MAX_PLANET_HEALTH = 100
    INITIAL_RESOURCES = 20
    TERRAFORM_COST = 10
    RESOURCE_GENERATION_RATE = 0.01  # Per terraformed quadrant per step
    WAVE_DURATION = 900  # steps (30 seconds at 30fps)
    HORROR_SPAWN_RATE_INCREASE = 0.2
    PATTERN_LENGTH_INCREASE_WAVE = 10
    
    GLYPHS = ['CIRCLE', 'TRIANGLE', 'SQUARE', 'CROSS']
    GLYPH_DIRECTIONS = {1: 'CIRCLE', 2: 'TRIANGLE', 3: 'SQUARE', 4: 'CROSS'} # map up/down/left/right to glyphs

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self._generate_stars()
        
        # self.reset() is called by the environment wrapper
        
    def _generate_stars(self):
        self.stars = []
        for i in range(200):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.uniform(0.5, 1.5)
            speed = size * 0.5
            self.stars.append({'pos': [x, y], 'size': size, 'speed': speed})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.planet_health = self.MAX_PLANET_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.terraformed_quadrants = [False, False, False, False] # Top-L, Top-R, Bot-L, Bot-R
        self.selected_quadrant = 0
        
        self.wave = 0
        self.wave_timer = self.WAVE_DURATION
        self.portals_to_spawn_in_wave = 0
        self.portal_spawn_timer = 0
        self.difficulty_spawn_rate_mod = 1.0
        self.difficulty_pattern_length = 2

        self.portals = []
        self.horrors = []
        self.particles = []
        
        self.defense_mode = False
        self.active_portal_idx = None
        self.selected_glyph_idx = 0 # 0=None, 1-4 for glyphs
        self.entered_defense_sequence = []

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        step_reward = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        step_reward += self._handle_input(movement, space_pressed, shift_pressed)
        step_reward += self._update_game_state()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.score += step_reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            step_reward -= 100 # Penalty for losing
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        reward = 0
        
        # Movement always selects something
        if movement in [1, 2]: # Up/Down
            self.selected_quadrant = 0 if movement == 1 else 2 # Top or Bottom half
        elif movement in [3, 4]: # Left/Right
            self.selected_quadrant = (self.selected_quadrant // 2) * 2 + (0 if movement == 3 else 1)
        
        if self.defense_mode:
            # In defense mode, movement selects glyphs, shift confirms
            if movement in self.GLYPH_DIRECTIONS:
                self.selected_glyph_idx = movement
            
            if shift_pressed and self.selected_glyph_idx != 0:
                # # Sound: UI_Confirm
                glyph = self.GLYPH_DIRECTIONS[self.selected_glyph_idx]
                self.entered_defense_sequence.append(glyph)
                
                # Check for defense success
                if self.active_portal_idx is not None and self.active_portal_idx < len(self.portals):
                    portal = self.portals[self.active_portal_idx]
                    
                    # Check if entered sequence matches portal pattern so far
                    pattern_part = portal['pattern'][:len(self.entered_defense_sequence)]
                    if self.entered_defense_sequence != pattern_part:
                        # # Sound: Defense_Fail
                        self.entered_defense_sequence = [] # Wrong glyph, reset sequence
                    
                    if self.entered_defense_sequence == portal['pattern']:
                        # # Sound: Defense_Success
                        reward += 5
                        portal['state'] = 'closing'
                        self._create_particles(portal['pos'], 30, self.COLOR_DEFENSE_UI, 2, 4)
                        self._exit_defense_mode()
        else:
            # In terraform mode, space terraforms
            if space_pressed:
                if not self.terraformed_quadrants[self.selected_quadrant] and self.resources >= self.TERRAFORM_COST:
                    # # Sound: Terraform_Action
                    self.resources -= self.TERRAFORM_COST
                    self.terraformed_quadrants[self.selected_quadrant] = True
                    reward += 0.1
                    self.score += 10 # Bonus score for terraforming
                    
                    # Find center of quadrant for particle effect
                    angle_start = (self.selected_quadrant * 90 - 45) * math.pi / 180
                    angle_end = ((self.selected_quadrant + 1) * 90 - 45) * math.pi / 180
                    mid_angle = (angle_start + angle_end) / 2
                    effect_pos = (
                        self.PLANET_CENTER[0] + self.PLANET_RADIUS * 0.7 * math.cos(mid_angle),
                        self.PLANET_CENTER[1] + self.PLANET_RADIUS * 0.7 * math.sin(mid_angle)
                    )
                    self._create_particles(effect_pos, 50, self.COLOR_TERRAFORM, 1, 3)
        return reward

    def _update_game_state(self):
        reward = 0
        
        # Resource generation
        generated = sum(self.terraformed_quadrants) * self.RESOURCE_GENERATION_RATE
        self.resources += generated
        reward += generated * 0.5
        
        # Wave management
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            if not self.portals and not self.horrors:
                # # Sound: Wave_Start
                self.wave += 1
                reward += 10 # Survived wave
                self.wave_timer = self.WAVE_DURATION
                self.portals_to_spawn_in_wave = self.wave
                self.portal_spawn_timer = 100 # Initial delay before first portal

                if self.wave > 0 and self.wave % 5 == 0:
                    self.difficulty_spawn_rate_mod += self.HORROR_SPAWN_RATE_INCREASE
                if self.wave > 0 and self.wave % self.PATTERN_LENGTH_INCREASE_WAVE == 0:
                    self.difficulty_pattern_length = min(8, self.difficulty_pattern_length + 1)

        # Portal spawning
        if self.portals_to_spawn_in_wave > 0:
            self.portal_spawn_timer -= 1
            if self.portal_spawn_timer <= 0:
                self._spawn_portal()
                self.portals_to_spawn_in_wave -= 1
                spawn_delay = (200 + random.randint(0, 100)) / self.difficulty_spawn_rate_mod
                self.portal_spawn_timer = int(spawn_delay)

        # Update entities
        reward += self._update_portals()
        self._update_horrors()
        self._update_particles()
        
        return reward

    def _spawn_portal(self):
        # # Sound: Portal_Open
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            pos = (random.randint(50, self.SCREEN_WIDTH - 50), 50)
        elif edge == 'bottom':
            pos = (random.randint(50, self.SCREEN_WIDTH - 50), self.SCREEN_HEIGHT - 50)
        elif edge == 'left':
            pos = (50, random.randint(50, self.SCREEN_HEIGHT - 50))
        else: # right
            pos = (self.SCREEN_WIDTH - 50, random.randint(50, self.SCREEN_HEIGHT - 50))
        
        pattern = random.choices(self.GLYPHS, k=self.difficulty_pattern_length)
        
        portal = {
            'pos': pos,
            'radius': 0,
            'max_radius': 30,
            'pattern': pattern,
            'state': 'opening', # opening -> pattern_display -> spawning -> closing
            'timer': 60, # Time for opening animation
            'glyph_flash_timer': 0
        }
        self.portals.append(portal)
        
        if not self.defense_mode:
            self._enter_defense_mode(len(self.portals) - 1)

    def _update_portals(self):
        reward = 0
        for i, portal in reversed(list(enumerate(self.portals))):
            portal['timer'] -= 1
            if portal['state'] == 'opening':
                portal['radius'] = min(portal['max_radius'], portal['max_radius'] * (1 - portal['timer'] / 60))
                if portal['timer'] <= 0:
                    portal['state'] = 'pattern_display'
                    portal['timer'] = 120 # Time to see pattern
                    portal['glyph_flash_timer'] = 30
            elif portal['state'] == 'pattern_display':
                portal['glyph_flash_timer'] = (portal['glyph_flash_timer'] - 1 + 30) % 30
                if portal['timer'] <= 0:
                    portal['state'] = 'spawning'
                    portal['timer'] = 90 # Time until horror spawns
            elif portal['state'] == 'spawning':
                if portal['timer'] <= 0:
                    # # Sound: Horror_Spawn
                    reward -= 2
                    self._spawn_horror(portal['pos'])
                    portal['state'] = 'closing'
                    if self.active_portal_idx == i:
                        self._exit_defense_mode()
            elif portal['state'] == 'closing':
                portal['radius'] -= 1
                if portal['radius'] <= 0:
                    self.portals.pop(i)
        return reward
        
    def _spawn_horror(self, pos):
        horror = {
            'pos': list(pos),
            'vel': [0, 0],
            'speed': random.uniform(1.0, 1.5),
            'size': 12,
            'anim_offset': random.uniform(0, 2 * math.pi)
        }
        self.horrors.append(horror)

    def _update_horrors(self):
        for horror in reversed(list(self.horrors)):
            # Move towards planet center
            direction = np.array(self.PLANET_CENTER) - np.array(horror['pos'])
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            
            horror['vel'] = direction * horror['speed']
            horror['pos'][0] += horror['vel'][0]
            horror['pos'][1] += horror['vel'][1]
            
            # Check collision with planet
            if dist < self.PLANET_RADIUS:
                # # Sound: Planet_Damage
                self.planet_health -= 10
                self.horrors.remove(horror)
                self._create_particles(horror['pos'], 40, self.COLOR_HORROR, 1.5, 3.5)
    
    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in reversed(list(self.particles)):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _enter_defense_mode(self, portal_idx):
        self.defense_mode = True
        self.active_portal_idx = portal_idx
        self.entered_defense_sequence = []
        self.selected_glyph_idx = 0
    
    def _exit_defense_mode(self):
        self.defense_mode = False
        self.active_portal_idx = None
        self.entered_defense_sequence = []
        
        # Find next available portal to defend against
        for i, p in enumerate(self.portals):
            if p['state'] in ['pattern_display', 'spawning']:
                self._enter_defense_mode(i)
                return

    def _check_termination(self):
        return self.planet_health <= 0

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
            "resources": self.resources,
            "planet_health": self.planet_health,
        }

    def _render_game(self):
        self._draw_starfield()
        self._draw_particles()
        self._draw_planet()
        self._draw_selector()
        self._draw_portals()
        self._draw_horrors()

    def _draw_starfield(self):
        for star in self.stars:
            star['pos'][0] = (star['pos'][0] - star['speed']) % self.SCREEN_WIDTH
            pos = (int(star['pos'][0]), int(star['pos'][1]))
            if star['size'] > 1:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(star['size']), self.COLOR_STAR)
            else:
                self.screen.set_at(pos, self.COLOR_STAR)

    def _draw_planet(self):
        # Draw terraformed quadrants
        for i in range(4):
            if self.terraformed_quadrants[i]:
                angle_start = (i * 90 - 45)
                angle_end = ((i + 1) * 90 - 45)
                rect = (self.PLANET_CENTER[0] - self.PLANET_RADIUS, self.PLANET_CENTER[1] - self.PLANET_RADIUS, self.PLANET_RADIUS * 2, self.PLANET_RADIUS * 2)
                pygame.draw.arc(self.screen, self.COLOR_TERRAFORM, rect, math.radians(angle_start), math.radians(angle_end), self.PLANET_RADIUS)

        # Draw glowing outline
        for i in range(5):
            alpha = 100 - i * 20
            color = (*self.COLOR_PLANET_OUTLINE, alpha)
            pygame.gfxdraw.aacircle(self.screen, self.PLANET_CENTER[0], self.PLANET_CENTER[1], self.PLANET_RADIUS + i, color)

    def _draw_selector(self):
        if not self.defense_mode:
            glow_color = (*self.COLOR_DEFENSE_UI, 100 + 50 * math.sin(self.steps * 0.2))
            angle_start = (self.selected_quadrant * 90 - 45)
            angle_end = ((self.selected_quadrant + 1) * 90 - 45)
            rect = (self.PLANET_CENTER[0] - self.PLANET_RADIUS, self.PLANET_CENTER[1] - self.PLANET_RADIUS, self.PLANET_RADIUS * 2, self.PLANET_RADIUS * 2)
            pygame.draw.arc(self.screen, glow_color, rect, math.radians(angle_start), math.radians(angle_end), 5)
    
    def _draw_portals(self):
        for i, portal in enumerate(self.portals):
            pos = (int(portal['pos'][0]), int(portal['pos'][1]))
            radius = int(portal['radius'])
            
            # Pulsing glow
            for r in range(radius, radius + 5):
                alpha = max(0, 150 - (r-radius)*30) * (0.75 + 0.25 * math.sin(self.steps * 0.1 + i))
                color = (*self.COLOR_PORTAL, int(alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r, color)
            
            # Swirling inner effect
            for j in range(3):
                offset_angle = self.steps * 0.05 + j * (2 * math.pi / 3)
                p1 = (pos[0] + math.cos(offset_angle) * radius * 0.8, pos[1] + math.sin(offset_angle) * radius * 0.8)
                p2 = (pos[0] - math.cos(offset_angle) * radius * 0.8, pos[1] - math.sin(offset_angle) * radius * 0.8)
                pygame.draw.aaline(self.screen, self.COLOR_PORTAL, p1, p2, 1)

            # Draw glyphs
            if portal['state'] == 'pattern_display' and portal['glyph_flash_timer'] > 15:
                for idx, glyph_type in enumerate(portal['pattern']):
                    angle = (idx / len(portal['pattern'])) * 2 * math.pi - math.pi/2
                    glyph_pos = (pos[0] + (radius + 20) * math.cos(angle), pos[1] + (radius + 20) * math.sin(angle))
                    self._draw_glyph(glyph_pos, glyph_type, self.COLOR_PORTAL, 8)
    
    def _draw_horrors(self):
        for horror in self.horrors:
            pos = (int(horror['pos'][0]), int(horror['pos'][1]))
            size = horror['size']
            
            # Pulsing body
            pulse = size * (1 + 0.1 * math.sin(self.steps * 0.2 + horror['anim_offset']))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse), self.COLOR_HORROR)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse), self.COLOR_HORROR)
            
            # Spikes
            for i in range(8):
                angle = i * (2 * math.pi / 8) + self.steps * 0.05
                start_pos = (pos[0] + math.cos(angle) * pulse, pos[1] + math.sin(angle) * pulse)
                end_pos = (pos[0] + math.cos(angle) * (pulse + size * 0.5), pos[1] + math.sin(angle) * (pulse + size * 0.5))
                pygame.draw.aaline(self.screen, self.COLOR_HORROR, start_pos, end_pos, 2)

    def _draw_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, min(255, p['life'] * 8))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def _draw_glyph(self, pos, glyph_type, color, size):
        x, y = int(pos[0]), int(pos[1])
        hs = size // 2
        if glyph_type == 'CIRCLE':
            pygame.gfxdraw.aacircle(self.screen, x, y, hs, color)
        elif glyph_type == 'SQUARE':
            pygame.gfxdraw.rectangle(self.screen, (x - hs, y - hs, size, size), color)
        elif glyph_type == 'TRIANGLE':
            points = [(x, y - hs), (x - hs, y + hs), (x + hs, y + hs)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif glyph_type == 'CROSS':
            pygame.draw.aaline(self.screen, color, (x - hs, y - hs), (x + hs, y + hs))
            pygame.draw.aaline(self.screen, color, (x + hs, y - hs), (x - hs, y + hs))

    def _render_ui(self):
        # Top left: Score and Wave
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (10, 35))

        # Top right: Resources
        resource_text = self.font_medium.render(f"RESOURCES: {int(self.resources)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - resource_text.get_width() - 10, 10))

        # Bottom center: Health bar
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        health_ratio = max(0, self.planet_health / self.MAX_PLANET_HEALTH)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_ratio), bar_height), border_radius=5)

        # Defense mode UI
        if self.defense_mode:
            # Show selected glyph
            ui_y = self.SCREEN_HEIGHT - 60
            if self.selected_glyph_idx != 0:
                glyph_type = self.GLYPH_DIRECTIONS[self.selected_glyph_idx]
                self._draw_glyph((self.SCREEN_WIDTH // 2, ui_y), glyph_type, self.COLOR_DEFENSE_UI, 20)
            
            # Show entered sequence
            seq_width = len(self.entered_defense_sequence) * 20
            start_x = self.SCREEN_WIDTH // 2 - seq_width // 2
            for i, glyph in enumerate(self.entered_defense_sequence):
                self._draw_glyph((start_x + i * 20, ui_y + 30), glyph, self.COLOR_DEFENSE_UI, 12)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("SYSTEM LOST", True, self.COLOR_HORROR)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # This code allows a human to play the game.
    # It will not run in a headless environment without a display.
    # To run, you might need to comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Cosmic Farm Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping for keyboard
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
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key
        
        env.clock.tick(30) # Limit to 30 FPS
            
    env.close()