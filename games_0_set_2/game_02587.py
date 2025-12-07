import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the crosshair. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric target practice. Hit all targets before time or ammo runs out. "
        "Risky shots near a target's edge grant bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    NUM_TARGETS = 15
    INITIAL_AMMO = 25
    TARGET_RADIUS = 12
    RISKY_SHOT_MARGIN = 4 # Hitting within this many pixels of the edge is a risky shot

    CROSSHAIR_SPEED = 8
    PROJECTILE_SPEED = 0.1  # Progress per frame, 10 frames to reach target

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GROUND = (40, 55, 71)
    COLOR_TARGET = (231, 76, 60)
    COLOR_TARGET_INNER = (192, 57, 43)
    COLOR_PROJECTILE = (241, 196, 15)
    COLOR_CROSSHAIR = (255, 255, 255)
    COLOR_CROSSHAIR_OUTLINE = (0, 0, 0)
    COLOR_HIT_SPARK = (46, 204, 113)
    COLOR_MISS_SPARK = (192, 57, 43)
    COLOR_TEXT = (236, 240, 241)
    COLOR_TEXT_SHADOW = (44, 62, 80)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 64)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.ammo = 0
        self.crosshair_pos = np.array([0.0, 0.0])
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.last_space_state = False
        self.fire_origin = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT + 20)
        self.targets_hit_count = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS
        self.ammo = self.INITIAL_AMMO
        self.crosshair_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.last_space_state = False
        
        self.projectiles.clear()
        self.particles.clear()
        
        self._generate_targets()
        
        return self._get_observation(), self._get_info()

    def _generate_targets(self):
        self.targets.clear()
        self.targets_hit_count = 0
        min_dist = 2.5 * self.TARGET_RADIUS # Min distance between target centers
        spawn_area = pygame.Rect(80, 50, self.SCREEN_WIDTH - 160, self.SCREEN_HEIGHT - 150)
        
        attempts = 0
        while len(self.targets) < self.NUM_TARGETS and attempts < 1000:
            attempts += 1
            pos = (
                self.np_random.integers(spawn_area.left, spawn_area.right),
                self.np_random.integers(spawn_area.top, spawn_area.bottom)
            )
            
            # Check for overlap
            too_close = False
            for target in self.targets:
                dist = math.hypot(pos[0] - target['pos'][0], pos[1] - target['pos'][1])
                if dist < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                self.targets.append({
                    'pos': pos,
                    'hit': False,
                    'hit_flash': 0, # Countdown timer for hit animation
                })

    def step(self, action):
        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        # shift_held is not used in this game
        
        if not self.game_over:
            self.steps += 1
            self.timer -= 1.0 / self.FPS

            # --- Handle Input ---
            self._handle_movement(movement)
            
            # Fire on button press (transition from not-pressed to pressed)
            if space_pressed and not self.last_space_state and self.ammo > 0:
                self._fire_projectile()
            self.last_space_state = space_pressed
            
            # --- Update Game State ---
            reward += self._update_projectiles()
            self._update_particles()
            self._update_targets()

            # --- Check Termination Conditions ---
            self.targets_hit_count = sum(1 for t in self.targets if t['hit'])
            all_targets_hit = self.targets_hit_count == len(self.targets)
            no_projectiles_in_flight = not self.projectiles
            
            if self.timer <= 0:
                terminated = True
            elif all_targets_hit:
                reward += 50.0  # Victory bonus
                terminated = True
            elif self.ammo == 0 and no_projectiles_in_flight:
                terminated = True
            
            if terminated:
                self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.crosshair_pos[1] -= self.CROSSHAIR_SPEED
        elif movement == 2: # Down
            self.crosshair_pos[1] += self.CROSSHAIR_SPEED
        elif movement == 3: # Left
            self.crosshair_pos[0] -= self.CROSSHAIR_SPEED
        elif movement == 4: # Right
            self.crosshair_pos[0] += self.CROSSHAIR_SPEED
        
        # Clamp crosshair to screen
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.SCREEN_WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.SCREEN_HEIGHT)

    def _fire_projectile(self):
        self.ammo -= 1
        self.projectiles.append({
            'start': np.array(self.fire_origin, dtype=float),
            'end': self.crosshair_pos.copy(),
            'progress': 0.0
        })

    def _update_projectiles(self):
        reward = 0.0
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj['progress'] += self.PROJECTILE_SPEED
            if proj['progress'] >= 1.0:
                projectiles_to_remove.append(proj)
                reward += self._check_projectile_hit(proj['end'])
        
        for proj in projectiles_to_remove:
            self.projectiles.remove(proj)
        
        return reward

    def _check_projectile_hit(self, pos):
        hit_target = False
        for target in self.targets:
            if not target['hit']:
                dist = math.hypot(pos[0] - target['pos'][0], pos[1] - target['pos'][1])
                if dist <= self.TARGET_RADIUS:
                    target['hit'] = True
                    target['hit_flash'] = 3 # Flash for 3 frames
                    hit_target = True
                    
                    # Base reward for hit
                    reward = 1.0
                    
                    # Bonus for risky "edge" shot
                    if self.TARGET_RADIUS - self.RISKY_SHOT_MARGIN < dist <= self.TARGET_RADIUS:
                        reward += 5.0
                    
                    self._create_particles(target['pos'], self.COLOR_HIT_SPARK, 30, 4.0)
                    return reward

        if not hit_target:
            self._create_particles(pos, self.COLOR_MISS_SPARK, 15, 2.0)
            return -2.0 # Penalty for missing a shot
        return 0.0

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(p)
        
        for p in particles_to_remove:
            self.particles.remove(p)

    def _update_targets(self):
        for target in self.targets:
            if target['hit_flash'] > 0:
                target['hit_flash'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground plane
        ground_points = [
            (self.SCREEN_WIDTH // 2, 20),
            (self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT // 2),
            (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20),
            (20, self.SCREEN_HEIGHT // 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, ground_points, self.COLOR_GROUND)
        pygame.gfxdraw.aapolygon(self.screen, ground_points, self.COLOR_GROUND)

        # Draw targets (draw stands first, then targets, for correct layering)
        for target in sorted(self.targets, key=lambda t: t['pos'][1]):
            if not target['hit']:
                # Draw stand
                stand_top = (target['pos'][0], target['pos'][1])
                stand_bottom = (target['pos'][0], target['pos'][1] + self.TARGET_RADIUS * 1.5)
                pygame.draw.aaline(self.screen, (0,0,0,50), stand_top, stand_bottom, 1)

        for target in sorted(self.targets, key=lambda t: t['pos'][1]):
            if target['hit_flash'] > 0:
                # Flash white on hit
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'][0]), int(target['pos'][1]), self.TARGET_RADIUS + 2, (255, 255, 255))
            elif not target['hit']:
                # Draw main target circle
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'][0]), int(target['pos'][1]), self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, int(target['pos'][0]), int(target['pos'][1]), self.TARGET_RADIUS, self.COLOR_TARGET)
                # Draw inner circle for detail
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'][0]), int(target['pos'][1]), self.TARGET_RADIUS // 2, self.COLOR_TARGET_INNER)
                pygame.gfxdraw.aacircle(self.screen, int(target['pos'][0]), int(target['pos'][1]), self.TARGET_RADIUS // 2, self.COLOR_TARGET_INNER)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            size = int(3 * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw projectiles
        for proj in self.projectiles:
            # Manual lerp: start + (end - start) * t
            current_pos = proj['start'] + (proj['end'] - proj['start']) * proj['progress']
            start_progress = max(0, proj['progress'] - 0.05)
            start_pos = proj['start'] + (proj['end'] - proj['start']) * start_progress
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start_pos, current_pos, 4)

        # Draw crosshair
        x, y = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
        size = 10
        # Outline
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR_OUTLINE, (x - size - 1, y), (x + size + 1, y), 3)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR_OUTLINE, (x, y - size - 1), (x, y + size + 1), 3)
        # Fill
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x - size, y), (x + size, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CROSSHAIR, (x, y - size), (x, y + size), 1)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            main_text = font.render(text, True, color)
            self.screen.blit(main_text, pos)

        # Draw UI elements
        draw_text(f"SCORE: {int(self.score)}", self.font_ui, self.COLOR_TEXT, (10, 10))
        draw_text(f"TIME: {max(0, int(self.timer))}", self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 10))
        ammo_color = self.COLOR_TARGET if self.ammo <= 5 else self.COLOR_TEXT
        draw_text(f"AMMO: {self.ammo}", self.font_ui, ammo_color, (10, 40))
        draw_text(f"TARGETS: {self.targets_hit_count}/{len(self.targets)}", self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 40))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            msg = "VICTORY!" if self.targets_hit_count == len(self.targets) else "GAME OVER"
            color = self.COLOR_HIT_SPARK if msg == "VICTORY!" else self.COLOR_MISS_SPARK
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "timer": self.timer,
            "targets_hit": self.targets_hit_count,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Target Practice")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # To keep track of space bar presses for firing
    last_space_state = False

    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_pressed = keys[pygame.K_SPACE]
        
        action = [movement, 1 if space_pressed else 0, 0] # shift not used
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Game ---")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(2000)
            print("--- Resetting Game ---")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()