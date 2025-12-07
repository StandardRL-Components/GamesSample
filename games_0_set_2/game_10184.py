import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:57:48.190939
# Source Brief: brief_00184.md
# Brief Index: 184
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth-action submarine game where the player uses card-drawn torpedoes
    to sink patterned enemy submarines in a deep-sea trench while avoiding sonar detection.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a stealth submarine, using special torpedoes to sink targets while avoiding sonar detection in a deep-sea trench."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to fire a torpedo. Hold shift to aim, and tap shift to cycle torpedo types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_SPEED = 3.0
    PLAYER_ROTATION_SPEED = 0.08
    TORPEDO_SPEED = 6.0
    MAX_STEPS = 1500
    NUM_ENEMIES = 3

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_TRENCH_LINE = (20, 40, 80)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 50)
    COLOR_VULNERABLE = (255, 255, 0)
    COLOR_TORPEDO = (255, 255, 255)
    COLOR_SONAR = (100, 150, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (50, 50, 50)
    COLOR_UI_BAR_FILL = (255, 180, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Game State Initialization ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_angle = 0.0
        self.aim_angle = 0.0
        self.enemies = []
        self.torpedoes = []
        self.particles = []
        self.sonar_pings = []
        self.torpedo_cards = []
        self.active_card_idx = 0
        self.steps = 0
        self.score = 0
        self.detection_level = 0.0
        self.sonar_base_freq = 150 # Steps between pings
        self.sonar_timer = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player
        self.player_pos = pygame.Vector2(
            self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50
        )
        self.player_angle = -math.pi / 2
        self.aim_angle = -math.pi / 2

        # Enemies
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(50, self.SCREEN_HEIGHT - 150)
                )
                # Ensure enemies are not too close to each other
                if all(pos.distance_to(e['pos']) > 100 for e in self.enemies):
                    break
            
            self.enemies.append({
                'pos': pos,
                'is_dead': False,
                'vulnerability_angle': self.np_random.choice([0, math.pi/2, math.pi, 3*math.pi/2]), # Top, Right, Bottom, Left
                'spawn_time': 0, # for initial fade-in effect
            })

        # Game State
        self.torpedoes = []
        self.particles = []
        self.sonar_pings = []
        
        self.steps = 0
        self.score = 0
        self.detection_level = 0.0
        self.sonar_timer = self.sonar_base_freq
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False

        # Torpedo Cards
        self.torpedo_cards = self.np_random.choice(['straight', 'curve_l', 'curve_r'], size=3).tolist()
        self.active_card_idx = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.step_reward = 0.0

        # --- Handle Input and Update State ---
        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_player(movement)
        self._update_torpedoes()
        self._update_sonar()
        self._update_particles()
        
        # --- Rewards & Termination ---
        # Small reward for survival, penalized by detection
        self.step_reward += 0.01 * (1 - self.detection_level / 100)

        terminated = self._check_termination()
        if terminated:
            if all(e['is_dead'] for e in self.enemies):
                self.step_reward += 100 # Victory bonus
            else:
                self.step_reward -= 100 # Detection penalty
        
        self.steps += 1
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated or truncated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Aiming (Shift)
        if shift_pressed:
            self.aim_angle += self.PLAYER_ROTATION_SPEED
            if self.aim_angle > 2 * math.pi: self.aim_angle -= 2 * math.pi
        
        # Cycle Torpedo Card (Shift - rising edge)
        if shift_pressed and not self.last_shift_held:
            self.active_card_idx = (self.active_card_idx + 1) % len(self.torpedo_cards)
        
        # Fire Torpedo (Space - rising edge)
        if space_pressed and not self.last_space_held:
            self._fire_torpedo()

        self.last_space_held = space_pressed
        self.last_shift_held = shift_pressed

    def _update_player(self, movement):
        vel = pygame.Vector2(0, 0)
        if movement == 1: vel.y = -1 # Up
        elif movement == 2: vel.y = 1 # Down
        elif movement == 3: vel.x = -1 # Left
        elif movement == 4: vel.x = 1 # Right

        if vel.length() > 0:
            vel.scale_to_length(self.PLAYER_SPEED)
            self.player_pos += vel
            # Smoothly turn player model to face movement direction
            target_angle = vel.angle_to(pygame.Vector2(1, 0)) * -math.pi / 180
            self.player_angle = self._lerp_angle(self.player_angle, target_angle, 0.15)
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.SCREEN_WIDTH - 10)
        self.player_pos.y = np.clip(self.player_pos.y, 10, self.SCREEN_HEIGHT - 10)

    def _fire_torpedo(self):
        # sound placeholder: # sfx_torpedo_launch()
        card_type = self.torpedo_cards[self.active_card_idx]
        
        start_pos = self.player_pos + pygame.Vector2(20, 0).rotate_rad(self.aim_angle)
        
        self.torpedoes.append({
            'pos': start_pos,
            'angle': self.aim_angle,
            'type': card_type,
            'lifespan': 120, # frames
            'turn_rate': 0.03 if card_type != 'straight' else 0,
            'turn_dir': 1 if card_type == 'curve_l' else -1
        })
        
        # Draw a new card
        self.torpedo_cards[self.active_card_idx] = self.np_random.choice(['straight', 'curve_l', 'curve_r'])
        
        # Create launch particles
        for _ in range(15):
            self.particles.append(self._create_particle(
                self.player_pos, life=20, size=3, speed_mult=0.5,
                color=(200, 200, 255), angle_spread=0.5, base_angle=self.aim_angle+math.pi
            ))

    def _update_torpedoes(self):
        for torp in reversed(self.torpedoes):
            # Update angle for curved torpedoes
            torp['angle'] += torp['turn_rate'] * torp['turn_dir']
            
            # Update position
            vel = pygame.Vector2(self.TORPEDO_SPEED, 0).rotate_rad(torp['angle'])
            torp['pos'] += vel
            
            # Trail particles
            if self.steps % 2 == 0:
                self.particles.append(self._create_particle(
                    torp['pos'], life=15, size=2, speed_mult=0.1, color=(200, 220, 255)
                ))

            # Check collisions
            hit = False
            for enemy in self.enemies:
                if not enemy['is_dead'] and enemy['pos'].distance_to(torp['pos']) < 15:
                    self._handle_torpedo_hit(torp, enemy)
                    hit = True
                    break
            
            # Wall collision or lifespan end
            if hit or torp['lifespan'] <= 0 or not self.screen.get_rect().collidepoint(torp['pos']):
                if not hit: # Create wall hit particles
                    # sfx_clank()
                    for _ in range(10): self.particles.append(self._create_particle(torp['pos'], color=(100, 100, 100)))
                self.torpedoes.remove(torp)
            else:
                torp['lifespan'] -= 1

    def _handle_torpedo_hit(self, torpedo, enemy):
        # Calculate impact angle relative to enemy
        impact_vec = (torpedo['pos'] - enemy['pos']).normalize()
        impact_angle = math.atan2(impact_vec.y, impact_vec.x)
        
        # Normalize angles to [0, 2pi)
        vuln_angle = (enemy['vulnerability_angle'] + 2 * math.pi) % (2 * math.pi)
        hit_angle = (impact_angle + math.pi + 2 * math.pi) % (2 * math.pi)

        angle_diff = abs(self._angle_diff(vuln_angle, hit_angle))
        
        if angle_diff < math.pi / 4: # 45 degree tolerance
            # Vulnerable hit!
            # sfx_explosion_large()
            self.score += 50
            self.step_reward += 10
            enemy['is_dead'] = True
            for _ in range(50): self.particles.append(self._create_particle(enemy['pos'], life=40, speed_mult=3, color=(255, 200, 50)))
        else:
            # Non-vulnerable hit
            # sfx_clank_heavy()
            self.score += 5
            self.step_reward += 1
            for _ in range(20): self.particles.append(self._create_particle(enemy['pos'], color=(150, 150, 150)))

    def _update_sonar(self):
        self.sonar_timer -= 1
        num_alive_enemies = sum(1 for e in self.enemies if not e['is_dead'])
        
        # Sonar frequency increases with fewer enemies
        current_sonar_freq = self.sonar_base_freq - (self.NUM_ENEMIES - num_alive_enemies) * 20

        if self.sonar_timer <= 0 and num_alive_enemies > 0:
            # sfx_sonar_ping()
            for enemy in self.enemies:
                if not enemy['is_dead']:
                    self.sonar_pings.append({
                        'pos': enemy['pos'],
                        'radius': 0,
                        'max_radius': 200,
                        'lifespan': 60,
                        'max_lifespan': 60,
                        'player_pinged': False
                    })
            self.sonar_timer = current_sonar_freq

        pinged_this_frame = False
        for ping in reversed(self.sonar_pings):
            ping['lifespan'] -= 1
            ping['radius'] = (1 - ping['lifespan'] / ping['max_lifespan']) * ping['max_radius']
            
            dist_to_player = self.player_pos.distance_to(ping['pos'])
            # Player is detected if within the expanding ring
            if not ping['player_pinged'] and abs(dist_to_player - ping['radius']) < 10:
                self.detection_level += 15
                self.step_reward -= 1.0
                ping['player_pinged'] = True
                pinged_this_frame = True

            if ping['lifespan'] <= 0:
                self.sonar_pings.remove(ping)
        
        if not pinged_this_frame:
            self.detection_level = max(0, self.detection_level - 0.1) # slow decay

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.detection_level >= 100:
            self.game_over = True
            return True
        if all(e['is_dead'] for e in self.enemies):
            self.game_over = True
            return True
        # MAX_STEPS handled by truncation
        return False
    
    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_sonar_pings()
        self._render_enemies()
        self._render_torpedoes()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(15):
            y = (self.steps * 0.1 + i * 50) % (self.SCREEN_HEIGHT + 50) - 50
            pygame.draw.line(self.screen, self.COLOR_TRENCH_LINE, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_sonar_pings(self):
        for ping in self.sonar_pings:
            alpha = int(255 * (ping['lifespan'] / ping['max_lifespan'])**2)
            color = self.COLOR_SONAR + (alpha,)
            pygame.gfxdraw.aacircle(self.screen, int(ping['pos'].x), int(ping['pos'].y), int(ping['radius']), color)

    def _render_enemies(self):
        for enemy in self.enemies:
            if not enemy['is_dead']:
                self._draw_triangle(enemy['pos'], math.pi, 15, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)
                # Draw vulnerability indicator
                vuln_indicator_pos = enemy['pos'] + pygame.Vector2(12, 0).rotate_rad(enemy['vulnerability_angle'])
                pygame.draw.circle(self.screen, self.COLOR_VULNERABLE, vuln_indicator_pos, 3)

    def _render_torpedoes(self):
        for torp in self.torpedoes:
            p1 = torp['pos']
            p2 = p1 - pygame.Vector2(10, 0).rotate_rad(torp['angle'])
            pygame.draw.line(self.screen, self.COLOR_TORPEDO, p1, p2, 2)
            pygame.draw.circle(self.screen, self.COLOR_TORPEDO, p1, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['life'] / p['max_life'])
            color = p['color']
            size = int(p['size'] * alpha)
            if size > 0:
                pygame.draw.circle(self.screen, color, p['pos'], size)

    def _render_player(self):
        self._draw_triangle(self.player_pos, self.player_angle, 12, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        # Draw aim indicator
        aim_end = self.player_pos + pygame.Vector2(40, 0).rotate_rad(self.aim_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_pos, aim_end, 1)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, aim_end, 3, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Detection Bar
        bar_width, bar_height = 150, 20
        bar_x, bar_y = self.SCREEN_WIDTH - bar_width - 10, 10
        fill_width = (self.detection_level / 100) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        det_text = self.font_small.render("DETECTION", True, self.COLOR_UI_TEXT)
        self.screen.blit(det_text, (bar_x + (bar_width - det_text.get_width()) // 2, bar_y + bar_height))

        # Torpedo Cards
        card_text = self.font_small.render("ORDNANCE", True, self.COLOR_UI_TEXT)
        self.screen.blit(card_text, (10, 10 + card_text.get_height()))

        for i, card in enumerate(self.torpedo_cards):
            is_active = (i == self.active_card_idx)
            box_rect = pygame.Rect(10, 40 + i * 35, 80, 30)
            border_color = self.COLOR_PLAYER if is_active else self.COLOR_UI_TEXT
            pygame.draw.rect(self.screen, border_color, box_rect, 2 if is_active else 1)
            self._draw_torpedo_icon(card, box_rect.center)
            
    def _draw_torpedo_icon(self, card_type, center):
        if card_type == 'straight':
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (center[0] - 10, center[1]), (center[0] + 10, center[1]), 2)
        elif card_type == 'curve_l':
            pygame.draw.arc(self.screen, self.COLOR_UI_TEXT, (center[0] - 10, center[1] - 10, 20, 20), math.pi, 2*math.pi, 2)
        elif card_type == 'curve_r':
            pygame.draw.arc(self.screen, self.COLOR_UI_TEXT, (center[0] - 10, center[1] - 10, 20, 20), 0, math.pi, 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "detection_level": self.detection_level,
            "enemies_left": sum(1 for e in self.enemies if not e['is_dead']),
        }

    def close(self):
        pygame.quit()

    # --- Helper Functions ---
    def _draw_triangle(self, pos, angle, size, color, glow_color):
        # Glow effect
        for i in range(size // 2, 0, -2):
            alpha = glow_color[3] * (1 - i / (size // 2))
            current_glow_color = glow_color[:3] + (int(alpha),)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), size + i, current_glow_color)
        
        # Main triangle body
        points = [
            pygame.Vector2(size, 0),
            pygame.Vector2(-size/2, -size/2),
            pygame.Vector2(-size/2, size/2),
        ]
        rotated_points = [p.rotate_rad(angle) + pos for p in points]
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)

    def _create_particle(self, pos, life=30, size=4, speed_mult=1.0, color=(255, 255, 255), angle_spread=2*math.pi, base_angle=0):
        angle = base_angle + self.np_random.uniform(-angle_spread / 2, angle_spread / 2)
        speed = self.np_random.uniform(0.5, 2.0) * speed_mult
        vel = pygame.Vector2(speed, 0).rotate_rad(angle)
        return {'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life, 'size': size, 'color': color}

    @staticmethod
    def _lerp_angle(a, b, t):
        diff = b - a
        if diff > math.pi: diff -= 2 * math.pi
        if diff < -math.pi: diff += 2 * math.pi
        return a + diff * t

    @staticmethod
    def _angle_diff(a1, a2):
        diff = a1 - a2
        return (diff + math.pi) % (2 * math.pi) - math.pi

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override Pygame screen for direct display
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Submarine Stealth")

    terminated = False
    truncated = False
    total_reward = 0
    
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    
    while not (terminated or truncated):
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        # --- Render to Display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            
    env.close()