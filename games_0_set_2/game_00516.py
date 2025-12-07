
# Generated: 2025-08-27T13:53:31.609131
# Source Brief: brief_00516.md
# Brief Index: 516

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    game_description = (
        "Survive hordes of procedurally generated zombies in an isometric 2D shooter arena. "
        "Eliminate all zombies to win, but watch your health and ammo!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game Constants ---
        self.ARENA_LOGICAL_SIZE = 250
        self.CENTER_X = self.screen_width // 2
        self.CENTER_Y = self.screen_height // 2 + 40

        self.MAX_STEPS = 1000
        self.STARTING_ZOMBIE_COUNT = 20

        # --- Entity Properties ---
        self.PLAYER_SPEED = 3.0
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 10
        self.PLAYER_SHOOT_COOLDOWN = 5  # steps
        self.PLAYER_SIZE = 6 # a radius for rendering

        self.ZOMBIE_SPEED = 0.75
        self.ZOMBIE_MAX_HEALTH = 50
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_SIZE = 5 # a radius for rendering

        self.BULLET_SPEED = 10.0
        self.BULLET_DAMAGE = 25
        self.BULLET_SIZE = 2

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_ARENA = (45, 55, 65)
        self.COLOR_ARENA_OUTLINE = (65, 75, 85)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ZOMBIE = (255, 60, 60)
        self.COLOR_BULLET = (255, 255, 200)
        self.COLOR_SHADOW = (0, 0, 0, 50)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_RED = (255, 80, 80)
        self.COLOR_AMMO = (80, 180, 255)

        # --- State variables ---
        self.player_pos = None
        self.player_health = 0
        self.player_ammo = 0
        self.player_shoot_cooldown_timer = 0
        self.player_is_reloading = False
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32)

        self.zombies = []
        self.bullets = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        # --- Initialize state ---
        self.reset()
        self.validate_implementation()

    def _to_iso(self, x, y):
        iso_x = self.CENTER_X + (x - y) * 1.2
        iso_y = self.CENTER_Y + (x + y) * 0.6
        return int(iso_x), int(iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.player_pos = np.array([self.ARENA_LOGICAL_SIZE / 2, self.ARENA_LOGICAL_SIZE / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_shoot_cooldown_timer = 0
        self.player_is_reloading = False
        self.player_last_move_dir = np.array([0, -1], dtype=np.float32)

        self.zombies = []
        for _ in range(self.STARTING_ZOMBIE_COUNT):
            self._spawn_zombie()

        self.bullets = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def _spawn_zombie(self):
        # Spawn on the edge of the arena
        edge = self.np_random.integers(0, 4)
        pos = np.zeros(2, dtype=np.float32)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.ARENA_LOGICAL_SIZE), 0])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.ARENA_LOGICAL_SIZE), self.ARENA_LOGICAL_SIZE])
        elif edge == 2: # Left
            pos = np.array([0, self.np_random.uniform(0, self.ARENA_LOGICAL_SIZE)])
        else: # Right
            pos = np.array([self.ARENA_LOGICAL_SIZE, self.np_random.uniform(0, self.ARENA_LOGICAL_SIZE)])

        self.zombies.append({
            "pos": pos,
            "health": self.ZOMBIE_MAX_HEALTH
        })

    def step(self, action):
        step_reward = 0
        self.game_over = self.player_health <= 0 or not self.zombies or self.steps >= self.MAX_STEPS

        if self.game_over:
            # If already game over, do nothing but return final state
            final_reward = 0
            if self.player_health <= 0 and self.win_message == "":
                final_reward = -100
                self.win_message = "YOU DIED"
            elif not self.zombies and self.win_message == "":
                final_reward = 100
                self.win_message = "VICTORY"
            elif self.steps >= self.MAX_STEPS and self.win_message == "":
                self.win_message = "TIME UP"

            self.score += final_reward
            return self._get_observation(), final_reward, True, False, self._get_info()

        # 1. HANDLE ACTIONS
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.player_is_reloading:
            self.player_ammo = self.PLAYER_MAX_AMMO
            self.player_is_reloading = False
            # sound: reload_finish.wav
        elif shift_held:
            self.player_is_reloading = True
            # sound: reload_start.wav
        else:
            # Movement
            move_vec = np.array([0, 0], dtype=np.float32)
            if movement == 1: move_vec[1] = -1  # Up
            elif movement == 2: move_vec[1] = 1   # Down
            elif movement == 3: move_vec[0] = -1  # Left
            elif movement == 4: move_vec[0] = 1   # Right

            if np.any(move_vec):
                norm = np.linalg.norm(move_vec)
                if norm > 0:
                    self.player_last_move_dir = move_vec / norm
                self.player_pos += self.player_last_move_dir * self.PLAYER_SPEED
                self.player_pos = np.clip(self.player_pos, 0, self.ARENA_LOGICAL_SIZE)

            # Shooting
            if space_held and self.player_ammo > 0 and self.player_shoot_cooldown_timer == 0:
                self.player_ammo -= 1
                self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
                self.bullets.append({"pos": self.player_pos.copy(), "dir": self.player_last_move_dir.copy()})
                self._create_particles(self.player_pos, self.COLOR_BULLET, 15, 4.0, 8) # Muzzle flash
                # sound: shoot.wav
            elif space_held and self.player_ammo == 0:
                step_reward -= 0.01 # Penalty for trying to shoot with no ammo
                # sound: empty_clip.wav

        # 2. UPDATE GAME STATE
        if self.player_shoot_cooldown_timer > 0:
            self.player_shoot_cooldown_timer -= 1

        # Update bullets
        for b in self.bullets:
            b['pos'] += b['dir'] * self.BULLET_SPEED
        self.bullets = [b for b in self.bullets if 0 < b['pos'][0] < self.ARENA_LOGICAL_SIZE and 0 < b['pos'][1] < self.ARENA_LOGICAL_SIZE]

        # Update zombies
        for z in self.zombies:
            direction_to_player = self.player_pos - z['pos']
            dist = np.linalg.norm(direction_to_player)
            if dist > 1:
                z['pos'] += (direction_to_player / dist) * self.ZOMBIE_SPEED

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # 3. HANDLE COLLISIONS
        # Bullets and Zombies
        zombies_to_remove_indices = set()
        bullets_to_remove_indices = set()
        for i, b in enumerate(self.bullets):
            if i in bullets_to_remove_indices: continue
            for j, z in enumerate(self.zombies):
                if j in zombies_to_remove_indices: continue
                if np.linalg.norm(b['pos'] - z['pos']) < self.ZOMBIE_SIZE + self.BULLET_SIZE:
                    z['health'] -= self.BULLET_DAMAGE
                    step_reward += 0.1
                    bullets_to_remove_indices.add(i)
                    self._create_particles(z['pos'], self.COLOR_ZOMBIE, 8, 2.0, 15)
                    # sound: zombie_hit.wav
                    if z['health'] <= 0:
                        zombies_to_remove_indices.add(j)
                        step_reward += 1.0
                        self._create_particles(z['pos'], self.COLOR_ZOMBIE, 30, 3.5, 25)
                        # sound: zombie_die.wav
                    break

        if zombies_to_remove_indices:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove_indices]
        if bullets_to_remove_indices:
            self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove_indices]

        # Player and Zombies
        for z in self.zombies:
            if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_SIZE + self.ZOMBIE_SIZE - 2:
                self.player_health -= self.ZOMBIE_DAMAGE
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 10, 2.5, 15)
                # sound: player_hurt.wav
                # Knockback
                knockback_dir = self.player_pos - z['pos']
                dist = np.linalg.norm(knockback_dir)
                if dist > 0:
                    self.player_pos += (knockback_dir / dist) * 10.0
                    self.player_pos = np.clip(self.player_pos, 0, self.ARENA_LOGICAL_SIZE)
        self.player_health = max(0, self.player_health)

        # 4. FINALIZE STEP
        self.steps += 1
        self.score += step_reward

        terminated = self.player_health <= 0 or not self.zombies or self.steps >= self.MAX_STEPS
        if terminated and self.win_message == "": # First frame of termination
            if self.player_health <= 0:
                step_reward -= 100
                self.win_message = "YOU DIED"
            elif not self.zombies:
                step_reward += 100
                self.win_message = "VICTORY"
            else: # Time up
                self.win_message = "TIME UP"
            self.game_over = True
            self.score += step_reward

        return self._get_observation(), step_reward, terminated, False, self._get_info()

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
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_left": len(self.zombies),
        }

    def _render_game(self):
        # Draw arena floor
        arena_points = [
            self._to_iso(0, 0),
            self._to_iso(self.ARENA_LOGICAL_SIZE, 0),
            self._to_iso(self.ARENA_LOGICAL_SIZE, self.ARENA_LOGICAL_SIZE),
            self._to_iso(0, self.ARENA_LOGICAL_SIZE)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, arena_points, self.COLOR_ARENA)
        pygame.gfxdraw.aapolygon(self.screen, arena_points, self.COLOR_ARENA_OUTLINE)

        # Z-sorting for correct isometric rendering
        render_list = []
        for z in self.zombies:
            render_list.append({"pos": z['pos'], "type": "zombie", "size": self.ZOMBIE_SIZE, "color": self.COLOR_ZOMBIE})
        render_list.append({"pos": self.player_pos, "type": "player", "size": self.PLAYER_SIZE, "color": self.COLOR_PLAYER})

        render_list.sort(key=lambda e: e['pos'][0] + e['pos'][1])

        # Render shadows first, then entities
        for item in render_list:
            self._render_shadow(item['pos'], item['size'])
        for item in render_list:
            self._render_entity(item['pos'], item['size'], item['color'])

        # Render bullets and particles on top
        for b in self.bullets:
            iso_pos = self._to_iso(b['pos'][0], b['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], self.BULLET_SIZE, self.COLOR_BULLET)

        for p in self.particles:
            iso_pos = self._to_iso(p['pos'][0], p['pos'][1])
            life_alpha = max(0, min(255, int(p['life'] / p['max_life'] * 255)))
            color = (*p['color'], life_alpha)
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(particle_surf, 2, 2, 2, color)
            self.screen.blit(particle_surf, (iso_pos[0] - 2, iso_pos[1] - 2))

    def _render_shadow(self, pos, size):
        iso_pos = self._to_iso(pos[0], pos[1])
        shadow_rect = pygame.Rect(0, 0, size * 2.5, size * 1.5)
        shadow_rect.center = (iso_pos[0], iso_pos[1] + size)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, *shadow_rect.size))
        self.screen.blit(shadow_surf, shadow_rect.topleft)

    def _render_entity(self, pos, size, color):
        iso_pos = self._to_iso(pos[0], pos[1])
        body_rect = pygame.Rect(0, 0, size * 2, size * 2)
        body_rect.center = (iso_pos[0], iso_pos[1] - size * 0.5)
        
        # Simple gradient for 3D effect
        top_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.ellipse(self.screen, color, body_rect)
        pygame.draw.arc(self.screen, top_color, body_rect, math.pi / 4, 3 * math.pi / 4, size)


    def _render_ui(self):
        # Health Bar
        bar_x, bar_y, bar_w, bar_h = 20, 20, 200, 20
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_RED, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))
        self._draw_text(f"HP: {int(self.player_health)}/{self.PLAYER_MAX_HEALTH}", (bar_x + bar_w + 10, bar_y + bar_h//2), center_y=True)

        # Ammo
        ammo_text = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        if self.player_is_reloading:
            ammo_text = "RELOADING..."
        self._draw_text(ammo_text, (self.screen_width // 2, self.screen_height - 20), center_x=True, center_y=True, color=self.COLOR_AMMO)

        # Zombie Count
        self._draw_text(f"Zombies: {len(self.zombies)}", (self.screen_width - 20, 20), align_right=True)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.win_message, (self.screen_width // 2, self.screen_height // 2), font=self.font_msg, center_x=True, center_y=True)

    def _draw_text(self, text, pos, font=None, color=None, center_x=False, center_y=False, align_right=False):
        if font is None: font = self.font_ui
        if color is None: color = self.COLOR_WHITE
        
        # Render with shadow for readability
        shadow_surf = font.render(text, True, tuple(c//2 for c in color))
        text_surf = font.render(text, True, color)

        text_rect = text_surf.get_rect()
        if center_x: text_rect.centerx = pos[0]
        elif align_right: text_rect.right = pos[0]
        else: text_rect.x = pos[0]

        if center_y: text_rect.centery = pos[1]
        else: text_rect.y = pos[1]

        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count, speed, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            life = self.np_random.integers(max_life // 2, max_life)
            self.particles.append({
                "pos": pos.copy() + self.np_random.uniform(-1, 1, 2),
                "vel": np.array([math.cos(angle) * vel_mag, math.sin(angle) * vel_mag]),
                "life": life,
                "max_life": life,
                "color": color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame window for human play ---
    pygame.display.set_caption("Isometric Zombie Shooter")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard inputs to MultiDiscrete action
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
        total_reward += reward
        
        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        env.clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
    env.close()