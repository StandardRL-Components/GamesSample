import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


# Set SDL to dummy mode for headless operation
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Import specific Gymnasium spaces and Pygame modules
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw

class GameEnv(gym.Env):
    """
    A tower-defense style game where the player must protect a central base
    from waves of incoming "paradox" enemies. Players spend resources to
    place defensive "protectors" and create temporary "rifts" to damage
    and slow enemies.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # --- User-facing metadata ---
    game_description = (
        "Defend your central base from incoming geometric 'paradoxes' by strategically placing defensive protectors and temporal rifts."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to place protectors on the screen. Press space to create a defensive rift around your base."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_BASE = (200, 200, 220)
        self.COLOR_BASE_GLOW = (100, 100, 200)
        self.COLOR_PROTECTOR = (0, 255, 150)
        self.COLOR_PROTECTOR_GLOW = (0, 150, 100)
        self.COLOR_PARADOX_TRIANGLE = (255, 80, 80)
        self.COLOR_PARADOX_SQUARE = (255, 120, 50)
        self.COLOR_PARADOX_PENTAGON = (255, 50, 150)
        self.COLOR_RIFT = (100, 150, 255)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_RED_FLASH = (255, 0, 0, 100)

        # --- Gameplay Parameters ---
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_RESOURCES = 100
        self.PROTECTOR_COST = 25
        self.RIFT_COST = 50
        self.PROTECTOR_RANGE = 150
        self.PROTECTOR_COOLDOWN = 20  # steps
        self.RIFT_DURATION = 300  # steps
        self.RIFT_RADIUS = 60
        self.RIFT_DPS = 0.1 # damage per step
        self.RIFT_SLOWDOWN = 0.5 # speed multiplier
        self.PARADOX_BASE_DAMAGE = 10
        self.PARADOX_INITIAL_SPAWN_PROB = 0.02
        self.PARADOX_SPAWN_RATE_INCREASE = 0.0005
        self.PARADOX_BASE_HEALTH = 5
        self.PARADOX_BASE_SPEED = 1.0

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action: [Movement (0-4), Place Rift (0-1), Unused (0-1)]
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.base_health = 0
        self.resources = 0
        self.paradox_spawn_prob = 0
        self.paradox_current_health = 0
        
        self.base_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.protectors = []
        self.paradoxes = []
        self.rifts = []
        self.projectiles = []
        self.particles = []
        self.background_stars = []

        self.screen_flash_timer = 0
        self.last_action = np.array([0, 0, 0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.paradox_spawn_prob = self.PARADOX_INITIAL_SPAWN_PROB
        self.paradox_current_health = self.PARADOX_BASE_HEALTH

        self.protectors.clear()
        self.paradoxes.clear()
        self.rifts.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.background_stars.clear()

        self.screen_flash_timer = 0
        self.last_action = np.array([0, 0, 0])

        for _ in range(100):
            self.background_stars.append({
                'pos': [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                'size': self.np_random.uniform(0.5, 1.5),
                'speed': self.np_random.uniform(0.05, 0.15)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # --- Handle Input (Trigger on press, not hold) ---
        self._handle_input(action)
        self.last_action = action

        # --- Update Game Logic ---
        self.steps += 1
        self.resources += 0.5 # Passive resource gain

        # Update difficulty
        self.paradox_spawn_prob += self.PARADOX_SPAWN_RATE_INCREASE
        if self.steps > 0 and self.steps % 500 == 0:
            self.paradox_current_health += 1

        # Spawn new paradoxes
        if self.np_random.random() < self.paradox_spawn_prob:
            self._spawn_paradox()

        # Update all entities
        reward += self._update_rifts()
        self._update_protectors()
        reward += self._update_projectiles()
        reward += self._update_paradoxes()
        self._update_particles()
        self._update_background()
        
        if self.screen_flash_timer > 0:
            self.screen_flash_timer -= 1
        
        # --- Calculate Reward & Termination ---
        reward += 0.01  # Survival reward

        terminated = self.base_health <= 0
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.base_health <= 0:
            reward -= 100.0 # Penalty for losing
        if truncated and self.base_health > 0:
            reward += 100.0  # Victory reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_btn, shift_btn = action
        
        # Protector placement (on press)
        if movement != 0 and self.last_action[0] == 0:
            if self.resources >= self.PROTECTOR_COST:
                self.resources -= self.PROTECTOR_COST
                self._spawn_protector(movement)
        
        # Rift placement (on press)
        if space_btn == 1 and self.last_action[1] == 0:
            if self.resources >= self.RIFT_COST:
                self.resources -= self.RIFT_COST
                self._spawn_rift()

    def _update_rifts(self):
        reward = 0
        for rift in self.rifts[:]:
            rift['lifetime'] -= 1
            rift['angle'] = (rift['angle'] + 2) % 360
            if rift['lifetime'] <= 0:
                self.rifts.remove(rift)
                continue
            
            for paradox in self.paradoxes:
                dist = math.hypot(paradox['pos'][0] - rift['pos'][0], paradox['pos'][1] - rift['pos'][1])
                if dist < self.RIFT_RADIUS:
                    paradox['health'] -= self.RIFT_DPS
                    paradox['slowed'] = True
                    paradox['hit_timer'] = 5 # Visual feedback
                    if paradox['health'] <= 0:
                         reward += 1.0 # Rift kill reward
        return reward

    def _update_protectors(self):
        for prot in self.protectors:
            prot['cooldown'] = max(0, prot['cooldown'] - 1)
            if prot['cooldown'] > 0:
                continue

            target = None
            min_dist = self.PROTECTOR_RANGE
            for paradox in self.paradoxes:
                dist = math.hypot(paradox['pos'][0] - prot['pos'][0], paradox['pos'][1] - prot['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = paradox
            
            if target:
                prot['cooldown'] = self.PROTECTOR_COOLDOWN
                direction = math.atan2(target['pos'][1] - prot['pos'][1], target['pos'][0] - prot['pos'][0])
                self.projectiles.append({
                    'pos': list(prot['pos']),
                    'vel': [math.cos(direction) * 8, math.sin(direction) * 8],
                })
                # Muzzle flash
                self._create_particle(prot['pos'], self.COLOR_PROJECTILE, count=1, speed=2)
        
    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]

            if not (0 < proj['pos'][0] < self.SCREEN_WIDTH and 0 < proj['pos'][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(proj)
                continue

            for paradox in self.paradoxes[:]:
                if math.hypot(proj['pos'][0] - paradox['pos'][0], proj['pos'][1] - paradox['pos'][1]) < 15:
                    paradox['health'] -= 1
                    paradox['hit_timer'] = 5
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    if paradox['health'] <= 0:
                        reward += 1.0 # Protector kill reward
                    break
        return reward

    def _update_paradoxes(self):
        reward = 0
        for p in self.paradoxes[:]:
            p['hit_timer'] = max(0, p['hit_timer'] - 1)
            p['rotation'] = (p['rotation'] + p['rot_speed']) % 360

            if p['health'] <= 0:
                self._create_particle(p['pos'], p['color'], count=30, speed=4)
                if p in self.paradoxes:
                    self.paradoxes.remove(p)
                continue

            speed = self.PARADOX_BASE_SPEED * self.RIFT_SLOWDOWN if p.get('slowed', False) else self.PARADOX_BASE_SPEED
            p['slowed'] = False # Reset slow status for next frame

            direction = math.atan2(p['target'][1] - p['pos'][1], p['target'][0] - p['pos'][0])
            p['pos'][0] += math.cos(direction) * speed
            p['pos'][1] += math.sin(direction) * speed

            if math.hypot(p['pos'][0] - self.base_pos[0], p['pos'][1] - self.base_pos[1]) < 25:
                self.base_health -= self.PARADOX_BASE_DAMAGE
                reward -= 5.0 # Penalty for hitting base
                self.screen_flash_timer = 10
                self._create_particle(p['pos'], self.COLOR_RED_FLASH, count=20, speed=3)
                if p in self.paradoxes:
                    self.paradoxes.remove(p)

        return reward

    def _update_particles(self):
        for part in self.particles[:]:
            part['pos'][0] += part['vel'][0]
            part['pos'][1] += part['vel'][1]
            part['lifetime'] -= 1
            if part['lifetime'] <= 0:
                self.particles.remove(part)
    
    def _update_background(self):
        for star in self.background_stars:
            star['pos'][1] += star['speed']
            if star['pos'][1] > self.SCREEN_HEIGHT:
                star['pos'][0] = self.np_random.uniform(0, self.SCREEN_WIDTH)
                star['pos'][1] = 0

    def _spawn_paradox(self):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
        elif edge == 1: # Right
            pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        elif edge == 2: # Bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
        else: # Left
            pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
        
        paradox_type = self.np_random.choice(['triangle', 'square', 'pentagon'])
        color_map = {
            'triangle': self.COLOR_PARADOX_TRIANGLE,
            'square': self.COLOR_PARADOX_SQUARE,
            'pentagon': self.COLOR_PARADOX_PENTAGON
        }
        
        target_offset_x = self.np_random.uniform(-20, 20)
        target_offset_y = self.np_random.uniform(-20, 20)
        target = [self.base_pos[0] + target_offset_x, self.base_pos[1] + target_offset_y]

        self.paradoxes.append({
            'pos': pos,
            'health': self.paradox_current_health,
            'type': paradox_type,
            'color': color_map[paradox_type],
            'target': target,
            'hit_timer': 0,
            'rotation': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-2, 2)
        })

    def _spawn_protector(self, location_code):
        margin = 30
        if location_code == 1: # Up
            pos = [self.np_random.uniform(margin, self.SCREEN_WIDTH - margin), margin]
        elif location_code == 2: # Down
            pos = [self.np_random.uniform(margin, self.SCREEN_WIDTH - margin), self.SCREEN_HEIGHT - margin]
        elif location_code == 3: # Left
            pos = [margin, self.np_random.uniform(margin, self.SCREEN_HEIGHT - margin)]
        else: # Right
            pos = [self.SCREEN_WIDTH - margin, self.np_random.uniform(margin, self.SCREEN_HEIGHT - margin)]
        
        self.protectors.append({
            'pos': pos,
            'cooldown': 0,
            'spawn_anim_timer': 10
        })
        self._create_particle(pos, self.COLOR_PROTECTOR, count=15, speed=2)
    
    def _spawn_rift(self):
        self.rifts.append({
            'pos': self.base_pos,
            'lifetime': self.RIFT_DURATION,
            'angle': 0
        })
        self._create_particle(self.base_pos, self.COLOR_RIFT, count=40, speed=3, lifetime=30)

    def _create_particle(self, pos, color, count, speed, lifetime=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(lifetime // 2, lifetime),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Stars
        for star in self.background_stars:
            pygame.draw.circle(self.screen, (50, 60, 80), star['pos'], star['size'])

        # Rifts
        for rift in self.rifts:
            alpha = int(100 + 100 * (rift['lifetime'] / self.RIFT_DURATION))
            radius = self.RIFT_RADIUS * (1 - rift['lifetime'] / (self.RIFT_DURATION * 1.5))
            self._draw_swirl(rift['pos'], radius, rift['angle'], self.COLOR_RIFT, alpha)

        # Base
        base_rect = pygame.Rect(0, 0, 40, 40)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        pygame.gfxdraw.rectangle(self.screen, base_rect, (*self.COLOR_BASE_GLOW, 100))

        # Protectors
        for prot in self.protectors:
            if prot['spawn_anim_timer'] > 0:
                prot['spawn_anim_timer'] -= 1
                radius = 10 * (1 - prot['spawn_anim_timer'] / 10.0)
            else:
                radius = 10
            
            pygame.gfxdraw.aacircle(self.screen, int(prot['pos'][0]), int(prot['pos'][1]), int(radius) + 4, (*self.COLOR_PROTECTOR_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, int(prot['pos'][0]), int(prot['pos'][1]), int(radius), self.COLOR_PROTECTOR)
            pygame.gfxdraw.aacircle(self.screen, int(prot['pos'][0]), int(prot['pos'][1]), int(radius), self.COLOR_PROTECTOR)

        # Projectiles
        for proj in self.projectiles:
            start_pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            end_pos = (int(proj['pos'][0] - proj['vel'][0] * 1.5), int(proj['pos'][1] - proj['vel'][1] * 1.5))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)
            
        # Paradoxes
        for p in self.paradoxes:
            color = self.COLOR_WHITE if p['hit_timer'] > 0 else p['color']
            self._draw_polygon(p['pos'], p['type'], 12, p['rotation'], color)
        
        # Particles
        for part in self.particles:
            alpha = 255 * (part['lifetime'] / 20.0)
            color = (*part['color'][:3], int(max(0, min(255, alpha))))
            size = part['size'] * (part['lifetime'] / 20.0)
            rect = pygame.Rect(part['pos'][0] - size/2, part['pos'][1] - size/2, size, size)
            try:
                pygame.draw.rect(self.screen, color, rect)
            except TypeError: # Handle potential invalid color from alpha calculation
                pass

        # Screen Flash
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_RED_FLASH[:3], self.screen_flash_timer * 10))
            self.screen.blit(flash_surface, (0, 0))

    def _draw_polygon(self, pos, p_type, radius, rotation, color):
        num_sides = {'triangle': 3, 'square': 4, 'pentagon': 5}[p_type]
        points = []
        for i in range(num_sides):
            angle = math.radians(rotation + (360 / num_sides) * i)
            x = pos[0] + radius * math.cos(angle)
            y = pos[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_swirl(self, pos, radius, start_angle, color, alpha):
        if radius <= 0: return
        for i in range(4):
            angle_offset = i * 90
            rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius * 2, radius * 2)
            start_rad = math.radians(start_angle + angle_offset)
            end_rad = math.radians(start_angle + angle_offset + 60)
            
            temp_surf = pygame.Surface((int(radius * 2), int(radius * 2)), pygame.SRCALPHA)
            try:
                pygame.draw.arc(temp_surf, (*color, alpha), (0, 0, int(radius*2), int(radius*2)), start_rad, end_rad, width=3)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
            except (ValueError, pygame.error): # Catch errors from radius being too small
                pass

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"Base Health: {max(0, int(self.base_health))}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Resources
        resource_text = self.font_ui.render(f"Resources: {int(self.resources)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (self.SCREEN_WIDTH - resource_text.get_width() - 10, 10))

        # Time
        time_text = self.font_ui.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, self.SCREEN_HEIGHT - time_text.get_height() - 10))

        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, self.SCREEN_HEIGHT - score_text.get_height() - 10))

        # Game Over
        if self.base_health <= 0:
            text = self.font_game_over.render("REALITY COLLAPSED", True, self.COLOR_PARADOX_TRIANGLE)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)
        elif self.steps >= self.MAX_STEPS:
            text = self.font_game_over.render("VICTORY", True, self.COLOR_PROTECTOR)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "paradox_count": len(self.paradoxes),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will create a Pygame window for rendering.
    os.environ.pop("SDL_VIDEODRIVER", None) # Use default video driver
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Pygame window for human interaction
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Paradox Protector")
    clock = pygame.time.Clock()

    # Store last action to detect key presses vs. holds
    last_action_manual = np.array([0, 0, 0])

    while running:
        if terminated or truncated:
            # Display final frame for 2 seconds then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False

        # --- Action mapping for human player ---
        keys = pygame.key.get_pressed()
        
        current_movement = 0
        if keys[pygame.K_UP]: current_movement = 1
        elif keys[pygame.K_DOWN]: current_movement = 2
        elif keys[pygame.K_LEFT]: current_movement = 3
        elif keys[pygame.K_RIGHT]: current_movement = 4
        
        current_space = 1 if keys[pygame.K_SPACE] else 0
        current_shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # The environment logic triggers actions on press, not hold.
        # So we only send a non-zero action for one frame.
        movement_action = current_movement if current_movement != 0 and last_action_manual[0] == 0 else 0
        space_action = current_space if current_space != 0 and last_action_manual[1] == 0 else 0
        
        # The environment doesn't use shift, but we include it for completeness
        action = [movement_action, space_action, current_shift]
        
        # Update last action state for next frame
        last_action_manual = np.array([current_movement, current_space, current_shift])

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling (to close the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.metadata['render_fps'])

    env.close()