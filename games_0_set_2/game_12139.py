import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:23:23.353447
# Source Brief: brief_02139.md
# Brief Index: 2139
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GeneCard:
    """Represents a single gene card with its effect."""
    def __init__(self):
        card_types = [
            ('TEMP_OPT', 'Optimal Temp', 0.05),
            ('HUMID_OPT', 'Optimal Humid', 0.05),
            ('TEMP_TOL', 'Temp Tolerance', 0.02),
            ('HUMID_TOL', 'Humid Tolerance', 0.02)
        ]
        self.type, self.name, self.base_value = random.choice(card_types)
        self.modifier = random.choice([-1, 1])
        self.value = self.base_value * self.modifier
        
        if self.modifier > 0:
            self.op_str = f"+{self.value:.2f}"
            self.color = (180, 255, 180)
        else:
            self.op_str = f"{self.value:.2f}"
            self.color = (255, 180, 180)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, angle, speed, radius, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.radius = radius
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.radius = max(0, self.radius * 0.98)
        return self.life > 0

    def draw(self, surface):
        if self.life <= 0: return
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color, alpha)
        
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
        surface.blit(temp_surf, (int(self.x - self.radius), int(self.y - self.radius)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Evolve a fractal creature to survive a harsh, changing environment. "
        "Apply gene cards to adapt its temperature and humidity tolerance."
    )
    user_guide = "Controls: ←→ to select a card, space to apply it, and shift to discard it."
    auto_advance = False

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_TURNS = 1000
    MAX_HEALTH = 100.0
    MAX_CARDS_IN_HAND = 4

    # --- COLORS ---
    COLOR_BG_NEUTRAL = (20, 30, 40)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_HIGH = (40, 220, 110)
    COLOR_HEALTH_MED = (240, 200, 50)
    COLOR_HEALTH_LOW = (230, 60, 60)
    COLOR_UI_BG = (10, 20, 30, 200)
    COLOR_CARD_BG = (40, 50, 70)
    COLOR_CARD_SELECTED = (255, 215, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 12)
        self.font_m = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Game state variables
        self.creature_health = self.MAX_HEALTH
        self.creature_attributes = {}
        self.visual_attributes = {}
        self.target_visual_attributes = {}
        
        self.environment = {}
        self.env_variance = 0.1
        
        self.card_hand = []
        self.selected_card_idx = 0
        
        self.particles = []
        self.feedback_deque = deque(maxlen=5)
        self.heat_wave_offset = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.creature_health = self.MAX_HEALTH
        self.creature_attributes = {
            'temp_opt': 0.5, 'humid_opt': 0.5,
            'temp_tol': 0.1, 'humid_tol': 0.1,
        }
        self.visual_attributes = {
            'angle': 45, 'length_ratio': 0.7, 'depth': 6, 'hue': 0.5
        }
        self.target_visual_attributes = self.visual_attributes.copy()
        
        self.environment = {'temp': 0.5, 'humid': 0.5}
        self.env_variance = 0.1
        
        self.card_hand = []
        self.selected_card_idx = 0
        while len(self.card_hand) < self.MAX_CARDS_IN_HAND:
            self.card_hand.append(GeneCard())
        
        self.particles = []
        self.feedback_deque.clear()
        self.feedback_deque.append((f"Episode Start. Survive {self.MAX_TURNS} turns.", self.COLOR_TEXT, 120))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 1.0  # Survival reward

        # 1. Handle player action
        self._handle_input(movement, space_held, shift_held)
        
        # 2. Update environment
        self._update_environment()

        # 3. Update creature health based on environment
        health_before = self.creature_health
        damage = self._calculate_damage()
        self.creature_health = max(0, self.creature_health - damage)
        if self.creature_health < health_before:
            reward -= 0.1
        
        # 4. Update creature visual targets based on attributes
        self._update_visual_targets()

        # 5. Check for special reward
        if self.env_variance > 0.5:
            reward += 5.0
            
        # 6. Check for termination
        terminated = False
        truncated = False
        if self.creature_health <= 0:
            terminated = True
            reward = -100.0
            self.feedback_deque.append(("Creature perished.", self.COLOR_HEALTH_LOW, 180))
            # sfx: game_over_sound
        elif self.steps >= self.MAX_TURNS:
            # Game won, but it's a truncation not termination
            truncated = True
            reward = 100.0
            self.feedback_deque.append(("Creature has achieved stable evolution!", self.COLOR_HEALTH_HIGH, 180))
            # sfx: victory_sound
        
        self.game_over = terminated or truncated
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if not self.card_hand:
            self.selected_card_idx = 0
            return

        # Handle selection
        if movement == 3: # Left
            self.selected_card_idx = (self.selected_card_idx - 1) % len(self.card_hand)
            # sfx: ui_select_blip
        elif movement == 4: # Right
            self.selected_card_idx = (self.selected_card_idx + 1) % len(self.card_hand)
            # sfx: ui_select_blip

        # Handle actions
        card_to_action = self.card_hand[self.selected_card_idx]
        action_taken = False

        if space_held: # Apply card
            self._apply_card(card_to_action)
            self.card_hand.pop(self.selected_card_idx)
            self.feedback_deque.append((f"Applied: {card_to_action.name} ({card_to_action.op_str})", card_to_action.color, 120))
            self._create_particles(self.WIDTH // 2, self.HEIGHT // 2, card_to_action.color, 30)
            # sfx: apply_card_chime
            action_taken = True
        elif shift_held: # Discard card
            self.card_hand.pop(self.selected_card_idx)
            self.feedback_deque.append((f"Discarded: {card_to_action.name}", self.COLOR_TEXT, 120))
            # sfx: discard_card_swoosh
            action_taken = True
        
        if action_taken:
            # Draw a new card to replace the used one
            self.card_hand.append(GeneCard())
            # Clamp selection to new hand size
            if self.selected_card_idx >= len(self.card_hand) and self.card_hand:
                self.selected_card_idx = len(self.card_hand) - 1

    def _apply_card(self, card):
        attr_map = {
            'TEMP_OPT': 'temp_opt', 'HUMID_OPT': 'humid_opt',
            'TEMP_TOL': 'temp_tol', 'HUMID_TOL': 'humid_tol'
        }
        attr = attr_map[card.type]
        self.creature_attributes[attr] += card.value
        # Clamp values to reasonable ranges
        self.creature_attributes[attr] = np.clip(self.creature_attributes[attr], 0, 1 if '_opt' in attr else 0.5)

    def _update_environment(self):
        if self.steps % 100 == 0 and self.steps > 0:
            self.env_variance = min(1.0, self.env_variance + 0.01)
        
        self.environment['temp'] = np.clip(self.np_random.normal(self.environment['temp'], self.env_variance * 0.2), 0, 1)
        self.environment['humid'] = np.clip(self.np_random.normal(self.environment['humid'], self.env_variance * 0.2), 0, 1)

    def _calculate_damage(self):
        temp_diff = abs(self.environment['temp'] - self.creature_attributes['temp_opt'])
        humid_diff = abs(self.environment['humid'] - self.creature_attributes['humid_opt'])
        
        temp_stress = max(0, temp_diff - self.creature_attributes['temp_tol'])
        humid_stress = max(0, humid_diff - self.creature_attributes['humid_tol'])
        
        # Damage scales with stress. The multiplier makes it significant.
        total_damage = (temp_stress + humid_stress) * 50
        return total_damage

    def _update_visual_targets(self):
        # Map creature attributes to visual parameters
        self.target_visual_attributes['angle'] = 15 + self.creature_attributes['temp_opt'] * 60
        self.target_visual_attributes['length_ratio'] = 0.65 + self.creature_attributes['humid_opt'] * 0.1
        self.target_visual_attributes['depth'] = 5 + int(self.creature_attributes['humid_tol'] * 20)
        self.target_visual_attributes['hue'] = self.creature_attributes['temp_tol'] * 2 # Map 0-0.5 tolerance to 0-1 hue range

    def _interpolate_visuals(self):
        # Smoothly transition current visuals towards target visuals
        for key in self.visual_attributes:
            current = self.visual_attributes[key]
            target = self.target_visual_attributes[key]
            self.visual_attributes[key] += (target - current) * 0.1 # Lerp factor

    def _get_observation(self):
        # Update animations
        self._interpolate_visuals()
        self.heat_wave_offset += 1

        # Main rendering pipeline
        self._render_background()
        self._render_environment_effects()
        self._render_creature()
        self._update_and_draw_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.creature_health,
            **self.creature_attributes,
            **self.environment
        }
        
    def _render_background(self):
        # Mix colors based on temperature (blue-red) and humidity (dark-light)
        temp_color = (self.environment['temp'] * 200, 0, (1 - self.environment['temp']) * 200)
        humid_factor = 0.5 + self.environment['humid'] * 0.5
        
        final_bg_color = (
            int(self.COLOR_BG_NEUTRAL[0] + temp_color[0] * humid_factor),
            int(self.COLOR_BG_NEUTRAL[1] + temp_color[1] * humid_factor),
            int(self.COLOR_BG_NEUTRAL[2] + temp_color[2] * humid_factor)
        )
        self.screen.fill(final_bg_color)
    
    def _render_environment_effects(self):
        # Heat waves for high temp
        if self.environment['temp'] > 0.7:
            alpha = int((self.environment['temp'] - 0.7) / 0.3 * 80)
            wave_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            for i in range(0, self.HEIGHT, 10):
                points = []
                for x in range(self.WIDTH):
                    y_offset = math.sin(x * 0.05 + self.heat_wave_offset * 0.1) * 3
                    points.append((x, i + y_offset))
                if len(points) > 1:
                    pygame.draw.lines(wave_surf, (255, 100, 100, alpha), False, points, 1)
            self.screen.blit(wave_surf, (0,0))
            
        # Ice crystals for low temp
        if self.environment['temp'] < 0.3:
            if random.random() < 0.3: # Spawn chance
                self.particles.append(Particle(
                    random.randint(0, self.WIDTH), 0, (180, 200, 255),
                    life=random.randint(80, 150), angle=math.pi/2, speed=random.uniform(0.5, 1.5),
                    radius=random.randint(2, 4), gravity=0.01
                ))

    def _render_creature(self):
        hue = self.visual_attributes['hue']
        health_ratio = self.creature_health / self.MAX_HEALTH
        
        # Color based on hue and saturation based on health
        color = pygame.Color(0)
        color.hsva = (hue * 360, health_ratio * 100, 100, 100)
        
        # Glow effect
        glow_color = pygame.Color(0)
        glow_color.hsva = (hue * 360, 80, 100, 100)
        
        start_pos = (self.WIDTH // 2, self.HEIGHT // 2 + 100)
        
        # Draw glow first
        self._recursive_fractal_draw(start_pos, -90, 80, self.visual_attributes['depth'], 10, glow_color, 0.5)
        # Draw main creature
        self._recursive_fractal_draw(start_pos, -90, 80, self.visual_attributes['depth'], 4, color, 1.0)
        
    def _recursive_fractal_draw(self, pos, angle, length, depth, width, color, length_multiplier):
        if depth <= 0 or length < 1 or width < 1:
            return
            
        rad_angle = math.radians(angle)
        end_pos = (pos[0] + length * math.cos(rad_angle), pos[1] + length * math.sin(rad_angle))
        
        pygame.draw.line(self.screen, color, pos, end_pos, int(width))
        
        branch_angle = self.visual_attributes['angle']
        length_ratio = self.visual_attributes['length_ratio']
        
        new_length = length * length_ratio * length_multiplier
        new_width = width * 0.8
        
        self._recursive_fractal_draw(end_pos, angle - branch_angle, new_length, depth - 1, new_width, color, length_multiplier)
        self._recursive_fractal_draw(end_pos, angle + branch_angle, new_length, depth - 1, new_width, color, length_multiplier)

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            if p.update():
                p.draw(self.screen)
                active_particles.append(p)
        self.particles = active_particles

    def _render_ui(self):
        # --- Cards ---
        card_w, card_h = 120, 70
        start_x = (self.WIDTH - len(self.card_hand) * (card_w + 10) + 10) / 2
        for i, card in enumerate(self.card_hand):
            is_selected = (i == self.selected_card_idx)
            card_x = start_x + i * (card_w + 10)
            card_y = self.HEIGHT - card_h - 10
            
            rect = pygame.Rect(card_x, card_y, card_w, card_h)
            pygame.draw.rect(self.screen, self.COLOR_CARD_BG, rect, border_radius=5)
            
            border_color = self.COLOR_CARD_SELECTED if is_selected else card.color
            border_width = 3 if is_selected else 1
            pygame.draw.rect(self.screen, border_color, rect, border_width, border_radius=5)
            
            name_surf = self.font_m.render(card.name, True, self.COLOR_TEXT)
            self.screen.blit(name_surf, (rect.centerx - name_surf.get_width()//2, rect.y + 8))
            
            val_surf = self.font_l.render(card.op_str, True, card.color)
            self.screen.blit(val_surf, (rect.centerx - val_surf.get_width()//2, rect.y + 30))

        # --- HUD ---
        # Health Bar
        health_ratio = self.creature_health / self.MAX_HEALTH
        health_color = self.COLOR_HEALTH_LOW if health_ratio < 0.3 else (self.COLOR_HEALTH_MED if health_ratio < 0.6 else self.COLOR_HEALTH_HIGH)
        bar_w = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_w + 10, 50), border_radius=5)
        pygame.draw.rect(self.screen, (50,50,50), (15, 35, bar_w, 20))
        pygame.draw.rect(self.screen, health_color, (15, 35, bar_w * health_ratio, 20))
        health_text = self.font_m.render(f"Health: {int(self.creature_health)}/{int(self.MAX_HEALTH)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 15))
        
        # Turn Counter
        turn_text = self.font_m.render(f"Turn: {self.steps}/{self.MAX_TURNS}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (self.WIDTH - turn_text.get_width() - 15, 15))

        # Score
        score_text = self.font_m.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 35))

        # Feedback Text
        for i, (text, color, life) in enumerate(list(self.feedback_deque)):
            alpha = min(255, int(life * 2.5))
            feedback_surf = self.font_s.render(text, True, (*color[:3], alpha))
            self.screen.blit(feedback_surf, (15, self.HEIGHT - 100 - i * 15))
            self.feedback_deque[i] = (text, color, life - 1)
        while self.feedback_deque and self.feedback_deque[0][2] <= 0:
            self.feedback_deque.popleft()

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(30, 60)
            radius = random.uniform(2, 6)
            self.particles.append(Particle(x, y, color, life, angle, speed, radius))
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It will not be run by the testing suite.
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # --- Manual Play Controls ---
        # Left/Right Arrow: Select card
        # Space: Apply selected card
        # Shift: Discard selected card
        
        running = True
        game_over_screen_timer = 120
        
        # Create a display for human play
        pygame.display.set_caption("Fractal Evolution")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        
        while running:
            action = [0, 0, 0] # no-op, released, released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        game_over_screen_timer = 120
                        
            if not env.game_over:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    action[0] = 3
                elif keys[pygame.K_RIGHT]:
                    action[0] = 4
                
                if keys[pygame.K_SPACE]:
                    action[1] = 1
                
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    action[2] = 1

                obs, reward, terminated, truncated, info = env.step(action)
            
            # Blit the observation to the display screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))

            if env.game_over:
                if game_over_screen_timer > 0:
                    s = pygame.Surface((GameEnv.WIDTH, GameEnv.HEIGHT), pygame.SRCALPHA)
                    s.fill((10, 20, 30, 200))
                    
                    win_text = "EVOLUTION COMPLETE" if env.steps >= GameEnv.MAX_TURNS else "EXTINCTION EVENT"
                    color = GameEnv.COLOR_HEALTH_HIGH if env.steps >= GameEnv.MAX_TURNS else GameEnv.COLOR_HEALTH_LOW
                    
                    title_font = pygame.font.SysFont("monospace", 48, bold=True)
                    sub_font = pygame.font.SysFont("monospace", 24, bold=True)

                    title_surf = title_font.render(win_text, True, color)
                    sub_surf = sub_font.render("Press 'R' to Restart", True, GameEnv.COLOR_TEXT)
                    
                    screen.blit(s, (0,0))
                    screen.blit(title_surf, (GameEnv.WIDTH/2 - title_surf.get_width()/2, GameEnv.HEIGHT/2 - 50))
                    screen.blit(sub_surf, (GameEnv.WIDTH/2 - sub_surf.get_width()/2, GameEnv.HEIGHT/2 + 20))
                    game_over_screen_timer -= 1
            
            pygame.display.flip()
            env.clock.tick(30) # Run at 30 FPS

    finally:
        if 'env' in locals():
            env.close()