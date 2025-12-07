import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:32:42.491932
# Source Brief: brief_01160.md
# Brief Index: 1160
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# --- Constants ---
# Screen
WIDTH, HEIGHT = 640, 400
# Colors
COLOR_BG = (15, 19, 25)
COLOR_GRID = (30, 35, 45)
COLOR_TEXT = (220, 220, 220)
COLOR_CURSOR = (255, 255, 255)
COLOR_PRISM = (255, 220, 0)
COLOR_PRISM_HEALTH_GOOD = (0, 200, 100)
COLOR_PRISM_HEALTH_BAD = (200, 50, 50)
COLOR_PROJECTILE = (255, 70, 70)
COLOR_TOWER_NEUTRAL = (150, 150, 150)
COLOR_TOWER_POSITIVE = (70, 150, 255)
COLOR_TOWER_NEGATIVE = (100, 255, 100)

# Game Parameters
MAX_STEPS = 1500
FPS = 30
CURSOR_SPEED = 10
PRISM_COUNT = 3
PRISM_MAX_HEALTH = 100
INITIAL_PROJECTILES_PER_WAVE = 3
INITIAL_PROJECTILE_SPEED = 1.5
TOWER_RADIUS = 15
TOWER_FIELD_STRENGTH = 25.0
TOWER_PLACEMENT_COOLDOWN = 10 # frames

# --- Helper Functions ---
def draw_glowing_circle(surface, color, center, radius, layers=5):
    """Draws a circle with a glowing aura."""
    temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        alpha = int(150 * (i / layers)**2)
        layer_color = (*color, alpha)
        pygame.gfxdraw.filled_circle(temp_surface, radius, radius, int(radius * (i / layers)), layer_color)
    surface.blit(temp_surface, (center[0] - radius, center[1] - radius))

def draw_rotated_polygon(surface, color, points, center, angle):
    """Draws a polygon rotated around a center point."""
    rotated_points = []
    for x, y in points:
        rad_angle = math.radians(angle)
        new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle) + center[0]
        new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle) + center[1]
        rotated_points.append((new_x, new_y))
    pygame.gfxdraw.aapolygon(surface, rotated_points, color)
    pygame.gfxdraw.filled_polygon(surface, rotated_points, color)

class Particle:
    def __init__(self, pos, vel, lifespan, color, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.radius = radius

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.95 # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            current_radius = int(self.radius * (self.lifespan / self.max_lifespan))
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_radius, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}
    
    game_description = (
        "Defend your prisms from incoming projectiles by strategically placing gravity-altering towers. "
        "Deflect projectiles and survive waves of increasing difficulty."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a tower and shift to cycle tower types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        # self._initialize_state() # Will be called by reset
        # self.validate_implementation() # Not needed in the final version
    
    def _initialize_state(self):
        """Initializes all game state variables. Called by __init__ and reset."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.prisms = [{'pos': pygame.Vector2(WIDTH - 40, HEIGHT // (PRISM_COUNT + 1) * (i + 1)), 
                        'health': PRISM_MAX_HEALTH, 'max_health': PRISM_MAX_HEALTH, 'radius': 15}
                       for i in range(PRISM_COUNT)]
        
        self.projectiles = []
        self.towers = []
        self.particles = []
        
        self.cursor_pos = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
        
        self.wave = 0
        self.wave_cleared = True
        self.projectiles_this_wave = INITIAL_PROJECTILES_PER_WAVE
        
        self.selected_tower_idx = 0
        self.unlocked_tower_types = ['NEUTRAL']
        
        self.last_space_held = False
        self.last_shift_held = False
        self.tower_placement_timer = 0
        
        self.events = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= MAX_STEPS
        
        if self.render_mode == "human":
            self._render_frame()
            self.clock.tick(self.metadata["render_fps"])

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, HEIGHT)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.unlocked_tower_types)
            # SFX: UI_Switch.wav

        # Place tower (on press)
        if space_held and not self.last_space_held and self.tower_placement_timer <= 0:
            self._place_tower()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _place_tower(self):
        is_valid_placement = True
        # Prevent placing on prisms
        for prism in self.prisms:
            if self.cursor_pos.distance_to(prism['pos']) < prism['radius'] + TOWER_RADIUS:
                is_valid_placement = False
                break
        # Prevent placing on other towers
        if is_valid_placement:
            for tower in self.towers:
                if self.cursor_pos.distance_to(tower['pos']) < TOWER_RADIUS * 2:
                    is_valid_placement = False
                    break
        
        if is_valid_placement:
            tower_type = self.unlocked_tower_types[self.selected_tower_idx]
            self.towers.append({'pos': self.cursor_pos.copy(), 'type': tower_type, 'radius': TOWER_RADIUS})
            self.tower_placement_timer = TOWER_PLACEMENT_COOLDOWN
            self.events.append("TOWER_PLACED")
            # SFX: Tower_Place.wav

    def _update_game_state(self):
        # Update timers
        if self.tower_placement_timer > 0:
            self.tower_placement_timer -= 1
            
        # Spawn wave
        if self.wave_cleared and not self.game_over:
            self._spawn_wave()
            
        # Update projectiles
        projectiles_to_remove = []
        for i, proj in enumerate(self.projectiles):
            # Apply magnetic forces
            total_force = pygame.Vector2(0, 0)
            for tower in self.towers:
                dist_vec = proj['pos'] - tower['pos']
                dist_sq = dist_vec.length_squared()
                if dist_sq > 1: # Avoid division by zero
                    force_magnitude = TOWER_FIELD_STRENGTH / dist_sq
                    polarity_interaction = 0
                    if tower['type'] == 'POSITIVE': polarity_interaction = 1
                    elif tower['type'] == 'NEGATIVE': polarity_interaction = -1
                    
                    total_force += dist_vec.normalize() * force_magnitude * polarity_interaction
            
            proj['vel'] += total_force
            # Cap speed
            speed = proj['vel'].length()
            max_speed = self.current_projectile_speed * 2.5
            if speed > max_speed:
                proj['vel'].scale_to_length(max_speed)
                
            proj['pos'] += proj['vel']
            proj['rotation'] = (proj['rotation'] + 5) % 360

            # Check collisions with prisms
            for prism in self.prisms:
                if prism['health'] > 0 and proj['pos'].distance_to(prism['pos']) < prism['radius']:
                    prism['health'] -= 25
                    self.events.append("PRISM_HIT")
                    if prism['health'] <= 0:
                        self.events.append("PRISM_DESTROYED")
                        # SFX: Explosion_Large.wav
                        for _ in range(50): self.particles.append(Particle(prism['pos'], pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)), random.randint(20, 40), COLOR_PRISM, random.randint(1, 4)))
                    else:
                        # SFX: Hit_Damage.wav
                        for _ in range(20): self.particles.append(Particle(proj['pos'], pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2)), random.randint(10, 20), COLOR_PROJECTILE, random.randint(1, 3)))
                    projectiles_to_remove.append(i)
                    break
            
            # Check out of bounds
            if not (0 < proj['pos'].x < WIDTH and 0 < proj['pos'].y < HEIGHT):
                if i not in projectiles_to_remove:
                    projectiles_to_remove.append(i)
                    self.events.append("DEFLECTED")
        
        # Remove projectiles
        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.projectiles[i]
            
        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
        # Check for wave clear
        if not self.projectiles and not self.wave_cleared:
            self.wave_cleared = True
            self.events.append("WAVE_CLEAR")

    def _spawn_wave(self):
        self.wave += 1
        self.wave_cleared = False
        
        # Unlock towers
        if self.wave >= 5 and 'POSITIVE' not in self.unlocked_tower_types: self.unlocked_tower_types.append('POSITIVE')
        if self.wave >= 10 and 'NEGATIVE' not in self.unlocked_tower_types: self.unlocked_tower_types.append('NEGATIVE')
        
        # Difficulty scaling
        self.current_projectile_speed = INITIAL_PROJECTILE_SPEED + 0.05 * (self.steps // 200)
        self.projectiles_this_wave = INITIAL_PROJECTILES_PER_WAVE + (self.wave - 1)
        
        for _ in range(self.projectiles_this_wave):
            start_y = random.uniform(50, HEIGHT - 50)
            target_prism = random.choice([p for p in self.prisms if p['health'] > 0] or self.prisms)
            
            direction = (target_prism['pos'] - pygame.Vector2(0, start_y)).normalize()
            
            self.projectiles.append({
                'pos': pygame.Vector2(0, start_y),
                'vel': direction * self.current_projectile_speed,
                'rotation': 0,
                'radius': 8
            })
        # SFX: Wave_Start.wav

    def _calculate_reward(self):
        reward = 0
        
        # Continuous reward for surviving prisms
        reward += 0.01 * sum(1 for p in self.prisms if p['health'] > 0)
        
        for event in self.events:
            if event == "DEFLECTED": reward += 1.0
            elif event == "PRISM_HIT": reward -= 2.0
            elif event == "PRISM_DESTROYED": reward -= 5.0
            elif event == "WAVE_CLEAR": reward += 50.0
            elif event == "TOWER_PLACED": reward -= 0.5
        
        self.events.clear()
        
        # Terminal rewards
        if self.steps >= MAX_STEPS and not self.game_over:
            reward += 100.0 # Victory bonus
        
        return reward

    def _check_termination(self):
        if all(p['health'] <= 0 for p in self.prisms):
            self.game_over = True
            
        return self.game_over

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _render_game(self):
        # Draw grid
        for x in range(0, WIDTH, 40): pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 40): pygame.draw.line(self.screen, COLOR_GRID, (0, y), (WIDTH, y))

        # Draw towers and fields
        for tower in self.towers:
            tower_color = {'NEUTRAL': COLOR_TOWER_NEUTRAL, 'POSITIVE': COLOR_TOWER_POSITIVE, 'NEGATIVE': COLOR_TOWER_NEGATIVE}[tower['type']]
            if tower['type'] != 'NEUTRAL':
                draw_glowing_circle(self.screen, tower_color, tower['pos'], TOWER_RADIUS * 6)
            pygame.gfxdraw.filled_circle(self.screen, int(tower['pos'].x), int(tower['pos'].y), tower['radius'], tower_color)
            pygame.gfxdraw.aacircle(self.screen, int(tower['pos'].x), int(tower['pos'].y), tower['radius'], (255,255,255))
    
        # Draw projectiles
        for proj in self.projectiles:
            points = [(0, -10), (6, 6), (-6, 6)]
            draw_rotated_polygon(self.screen, COLOR_PROJECTILE, points, proj['pos'], proj['rotation'])
    
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw prisms
        for prism in self.prisms:
            if prism['health'] > 0:
                # Prism body
                pygame.gfxdraw.filled_circle(self.screen, int(prism['pos'].x), int(prism['pos'].y), prism['radius'], COLOR_PRISM)
                pygame.gfxdraw.aacircle(self.screen, int(prism['pos'].x), int(prism['pos'].y), prism['radius'], (255,255,255))
                
                # Health bar
                bar_width = 40
                bar_height = 5
                bar_pos = (prism['pos'].x - bar_width // 2, prism['pos'].y - prism['radius'] - 15)
                health_ratio = prism['health'] / prism['max_health']
                
                pygame.draw.rect(self.screen, COLOR_PRISM_HEALTH_BAD, (*bar_pos, bar_width, bar_height))
                pygame.draw.rect(self.screen, COLOR_PRISM_HEALTH_GOOD, (*bar_pos, int(bar_width * health_ratio), bar_height))

        # Draw cursor
        cursor_color = {'NEUTRAL': COLOR_TOWER_NEUTRAL, 'POSITIVE': COLOR_TOWER_POSITIVE, 'NEGATIVE': COLOR_TOWER_NEGATIVE}[self.unlocked_tower_types[self.selected_tower_idx]]
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), TOWER_RADIUS, (*cursor_color, 150))
        pygame.draw.line(self.screen, COLOR_CURSOR, (self.cursor_pos.x - 5, self.cursor_pos.y), (self.cursor_pos.x + 5, self.cursor_pos.y), 1)
        pygame.draw.line(self.screen, COLOR_CURSOR, (self.cursor_pos.x, self.cursor_pos.y - 5), (self.cursor_pos.x, self.cursor_pos.y + 5), 1)
        
    def _render_ui(self):
        # Score, Steps, Wave
        score_text = self.font_small.render(f"SCORE: {int(self.score):,}", True, COLOR_TEXT)
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{MAX_STEPS}", True, COLOR_TEXT)
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))
        self.screen.blit(wave_text, (10, 50))
        
        # Selected Tower
        selected_type = self.unlocked_tower_types[self.selected_tower_idx]
        tower_color = {'NEUTRAL': COLOR_TOWER_NEUTRAL, 'POSITIVE': COLOR_TOWER_POSITIVE, 'NEGATIVE': COLOR_TOWER_NEGATIVE}[selected_type]
        tower_text = self.font_large.render(f"Selected: {selected_type}", True, tower_color)
        text_rect = tower_text.get_rect(center=(WIDTH // 2, HEIGHT - 30))
        self.screen.blit(tower_text, text_rect)
        
        # Game Over
        if self.game_over or self.steps >= MAX_STEPS:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            survived = all(p['health'] > 0 for p in self.prisms)
            end_text_str = "VICTORY" if survived else "DEFENSES FAILED"
            end_color = COLOR_PRISM_HEALTH_GOOD if survived else COLOR_PRISM_HEALTH_BAD
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

        # human mode
        if not hasattr(self, 'human_screen'):
            pygame.display.init()
            pygame.display.set_caption("Prism Defense")
            self.human_screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Blit the internal screen to the display screen
        self.human_screen.blit(self.screen, (0, 0))
        pygame.event.pump()
        pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play / Demonstration ---
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Use a deque to store recent actions for smoother control
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print("\n--- Manual Control ---")
    print("Arrows: Move cursor")
    print("Space: Place tower")
    print("Shift: Cycle tower type")
    print("Q: Quit")
    
    while not done:
        # Action defaults
        movement = 0
        space_held = 0
        shift_held = 0
        
        # Poll events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

        # Check pressed keys
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize up/down/left/right in that order
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render() # Manually call render for human mode
        
        if terminated or truncated:
            done = True
            print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause to see final screen

    env.close()