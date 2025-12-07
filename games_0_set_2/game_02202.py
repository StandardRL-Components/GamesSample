
# Generated: 2025-08-27T19:35:48.626592
# Source Brief: brief_02202.md
# Brief Index: 2202

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Escape a procedurally generated haunted mansion as a shadow creature,
    battling spectral enemies to reach the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ↑ to jump, ←→ to run. Press space to attack."
    )
    game_description = (
        "A fast-paced platformer set in a haunted mansion. Escape by reaching the exit, "
        "but beware of spectral enemies and deadly traps."
    )

    # Frame advance behavior
    auto_advance = True

    # --- Constants ---
    # Screen and Level
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LEVEL_WIDTH = 5000
    LEVEL_GROUND_Y = 350
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_DECOR = (25, 20, 40)
    COLOR_PLATFORM = (40, 30, 55)
    COLOR_SPIKE = (150, 150, 170)
    COLOR_PLAYER = (5, 5, 15)
    COLOR_PLAYER_EYES = (180, 255, 255)
    COLOR_EXIT_GLOW = (100, 150, 255)
    COLOR_ATTACK = (255, 255, 255)
    COLOR_HEALTH = (200, 40, 40)
    COLOR_HEALTH_BG = (80, 20, 20)
    COLOR_TEXT = (220, 220, 230)

    # Physics
    GRAVITY = 0.6
    FRICTION = 0.85
    PLAYER_ACCEL = 1.2
    PLAYER_MAX_SPEED = 8
    JUMP_STRENGTH = -12

    # Player
    PLAYER_MAX_HEALTH = 5
    PLAYER_ATTACK_COOLDOWN = 15  # frames
    PLAYER_ATTACK_DURATION = 5
    PLAYER_ATTACK_RANGE = 50
    PLAYER_INVINCIBILITY_DURATION = 45 # frames

    # Enemy
    ENEMY_CHARGE_RADIUS = 250
    ENEMY_RESPAWN_TIME = 120 # frames

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.player = {}
        self.platforms = []
        self.spikes = []
        self.enemies = []
        self.particles = []
        self.decorations = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.enemy_speed_multiplier = 1.0

        # This will be initialized in reset()
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.enemy_speed_multiplier = 1.0

        self.player = {
            "pos": np.array([150.0, self.LEVEL_GROUND_Y - 50]),
            "vel": np.array([0.0, 0.0]),
            "size": np.array([20, 40]),
            "health": self.PLAYER_MAX_HEALTH,
            "on_ground": False,
            "attack_timer": 0,
            "attack_cooldown": 0,
            "invincibility_timer": 0,
            "facing_right": True
        }

        self.enemies.clear()
        self.particles.clear()
        self._generate_level()
        self.camera_x = 0

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms.clear()
        self.spikes.clear()
        self.decorations.clear()

        # Create a continuous floor with variations
        x = -self.SCREEN_WIDTH
        y = self.LEVEL_GROUND_Y
        while x < self.LEVEL_WIDTH + self.SCREEN_WIDTH:
            width = self.np_random.integers(200, 500)
            self.platforms.append(pygame.Rect(x, y, width, self.SCREEN_HEIGHT - y))

            # Add potential enemies and spikes on this platform
            if x > 500 and x < self.LEVEL_WIDTH - 200:
                if self.np_random.random() < 0.5:
                    self._spawn_enemy(x + width / 2, y - 20)
                if self.np_random.random() < 0.2:
                    self.spikes.append(pygame.Rect(x + self.np_random.integers(50, width-50), y - 10, 30, 10))

            # Create next segment (gap or step)
            gap = self.np_random.integers(0, 100) if self.np_random.random() < 0.7 else 0
            x += width + gap
            if self.np_random.random() < 0.4:
                y += self.np_random.integers(-50, 51)
                y = np.clip(y, self.LEVEL_GROUND_Y - 100, self.LEVEL_GROUND_Y + 100)

        # Add background decorations
        for _ in range(50):
            dx = self.np_random.integers(0, self.LEVEL_WIDTH)
            dy = self.np_random.integers(50, self.LEVEL_GROUND_Y - 100)
            dw = self.np_random.integers(20, 80)
            dh = self.np_random.integers(40, 150)
            self.decorations.append(pygame.Rect(dx, dy, dw, dh)) # windows/arches

    def _spawn_enemy(self, x, y):
        enemy_type = self.np_random.choice(['wisp', 'shade', 'spectre'])
        self.enemies.append({
            "pos": np.array([x, y]),
            "vel": np.array([0.0, 0.0]),
            "size": np.array([25, 25]),
            "type": enemy_type,
            "health": 1,
            "on_ground": False,
            "state_timer": self.np_random.integers(60, 120),
            "patrol_center": x,
            "facing_right": self.np_random.choice([True, False]),
            "death_timer": 0
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Update game logic ---
        self._handle_input(action)
        prev_x = self.player["pos"][0]
        self._update_player()
        reward += (self.player["pos"][0] - prev_x) * 0.01 # Reward for moving right

        damage_taken, enemies_killed = self._update_enemies()

        reward += enemies_killed * 1.0
        self.score += enemies_killed

        if damage_taken:
            reward -= 0.1
            self._create_particles(self.player["pos"] + self.player["size"]/2, 10, self.COLOR_HEALTH)

        self._update_particles()
        self._update_difficulty()

        # --- Check for termination ---
        terminated = False
        if self.player["health"] <= 0:
            reward = -10.0
            self.game_over = True
            terminated = True
        elif self.player["pos"][0] > self.LEVEL_WIDTH:
            reward = 10.0
            self.game_over = True
            self.game_won = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 3:  # Left
            self.player["vel"][0] -= self.PLAYER_ACCEL
            self.player["facing_right"] = False
        if movement == 4:  # Right
            self.player["vel"][0] += self.PLAYER_ACCEL
            self.player["facing_right"] = True

        # Jump
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"][1] = self.JUMP_STRENGTH
            self.player["on_ground"] = False
            # sfx: jump

        # Attack
        if space_held and self.player["attack_cooldown"] <= 0:
            self.player["attack_timer"] = self.PLAYER_ATTACK_DURATION
            self.player["attack_cooldown"] = self.PLAYER_ATTACK_COOLDOWN
            # sfx: attack_swing

    def _update_player(self):
        p = self.player

        # Update timers
        if p["attack_timer"] > 0: p["attack_timer"] -= 1
        if p["attack_cooldown"] > 0: p["attack_cooldown"] -= 1
        if p["invincibility_timer"] > 0: p["invincibility_timer"] -= 1

        # Apply physics
        p["vel"][1] += self.GRAVITY
        p["vel"][0] *= self.FRICTION
        p["vel"][0] = np.clip(p["vel"][0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Move and collide
        p["on_ground"] = False
        p["pos"][0] += p["vel"][0]
        self._collide_with_platforms(p, 'horizontal')

        p["pos"][1] += p["vel"][1]
        self._collide_with_platforms(p, 'vertical')
        
        # Spike collision
        if p["invincibility_timer"] <= 0:
            player_rect = pygame.Rect(*p["pos"], *p["size"])
            for spike in self.spikes:
                if player_rect.colliderect(spike):
                    p["health"] -= 1
                    p["invincibility_timer"] = self.PLAYER_INVINCIBILITY_DURATION
                    p["vel"][1] = self.JUMP_STRENGTH / 2 # bounce up
                    # sfx: player_hurt
                    break
        
        # World bounds
        p["pos"][0] = np.clip(p["pos"][0], 0, self.LEVEL_WIDTH + 100)
        if p["pos"][1] > self.SCREEN_HEIGHT + 100: # Fell out of world
            p["health"] = 0


    def _collide_with_platforms(self, entity, axis):
        entity_rect = pygame.Rect(*entity["pos"], *entity["size"])
        for plat in self.platforms:
            if entity_rect.colliderect(plat):
                if axis == 'horizontal':
                    if entity["vel"][0] > 0:
                        entity_rect.right = plat.left
                    elif entity["vel"][0] < 0:
                        entity_rect.left = plat.right
                    entity["vel"][0] = 0
                elif axis == 'vertical':
                    if entity["vel"][1] > 0:
                        entity_rect.bottom = plat.top
                        entity["on_ground"] = True
                    elif entity["vel"][1] < 0:
                        entity_rect.top = plat.bottom
                    entity["vel"][1] = 0
                entity["pos"] = np.array(entity_rect.topleft, dtype=float)


    def _update_enemies(self):
        damage_taken = False
        enemies_killed = 0
        player_rect = pygame.Rect(*self.player["pos"], *self.player["size"])

        # Player attack hitbox
        attack_rect = None
        if self.player["attack_timer"] > 0:
            x_offset = 0 if self.player["facing_right"] else -self.PLAYER_ATTACK_RANGE
            attack_rect = pygame.Rect(
                self.player["pos"][0] + x_offset,
                self.player["pos"][1],
                self.player["size"][0] + self.PLAYER_ATTACK_RANGE,
                self.player["size"][1]
            )

        for e in self.enemies:
            if e["health"] <= 0:
                if e["death_timer"] > 0:
                    e["death_timer"] -= 1
                else: # Respawn
                    self._spawn_enemy(e["patrol_center"], self.LEVEL_GROUND_Y - 20)
                continue # Skip logic for dead enemies

            # AI
            dist_to_player = np.linalg.norm(self.player["pos"] - e["pos"])
            self._enemy_ai(e, dist_to_player)

            # Physics
            e["vel"][1] += self.GRAVITY
            e["pos"] += e["vel"] * self.enemy_speed_multiplier
            self._collide_with_platforms(e, 'vertical')
            self._collide_with_platforms(e, 'horizontal')

            # Collision check
            enemy_rect = pygame.Rect(*e["pos"], *e["size"])

            # Player hits enemy
            if attack_rect and enemy_rect.colliderect(attack_rect):
                e["health"] = 0
                e["death_timer"] = self.ENEMY_RESPAWN_TIME
                enemies_killed += 1
                self._create_particles(e["pos"] + e["size"]/2, 20, (200, 200, 255))
                # sfx: enemy_death
                continue

            # Enemy hits player
            if enemy_rect.colliderect(player_rect) and self.player["invincibility_timer"] <= 0:
                self.player["health"] -= 1
                self.player["invincibility_timer"] = self.PLAYER_INVINCIBILITY_DURATION
                damage_taken = True
                # sfx: player_hurt
                # Knockback
                dx = self.player["pos"][0] - e["pos"][0]
                self.player["vel"][0] = 5 * np.sign(dx) if dx != 0 else 5
                self.player["vel"][1] = -5

        # Remove fully dead enemies and add them back to the pool
        self.enemies = [e for e in self.enemies if not (e["health"] <= 0 and e["death_timer"] <= 0)]
        return damage_taken, enemies_killed

    def _enemy_ai(self, e, dist_to_player):
        e["state_timer"] -= 1
        
        if e["type"] == 'wisp':
            e["vel"][0] = 1.5 if e["facing_right"] else -1.5
            if abs(e["pos"][0] - e["patrol_center"]) > 100 or e["state_timer"] <= 0:
                e["facing_right"] = not e["facing_right"]
                e["state_timer"] = self.np_random.integers(120, 240)

        elif e["type"] == 'shade':
            if e["on_ground"] and e["state_timer"] <= 0:
                e["vel"][1] = self.JUMP_STRENGTH * 0.7
                e["vel"][0] = self.np_random.uniform(-2, 2)
                e["state_timer"] = self.np_random.integers(90, 180)
            if e["on_ground"]:
                e["vel"][0] *= 0.8

        elif e["type"] == 'spectre':
            if dist_to_player < self.ENEMY_CHARGE_RADIUS:
                # Charge player
                direction = self.player["pos"][0] - e["pos"][0]
                e["vel"][0] = 3 * np.sign(direction) if direction != 0 else 0
            else:
                # Patrol
                e["vel"][0] = 0.5 if e["facing_right"] else -0.5
                if abs(e["pos"][0] - e["patrol_center"]) > 150 or e["state_timer"] <= 0:
                    e["facing_right"] = not e["facing_right"]
                    e["state_timer"] = self.np_random.integers(120, 240)


    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]),
                "life": self.np_random.integers(10, 30),
                "color": color
            })
            
    def _update_difficulty(self):
        self.enemy_speed_multiplier = 1.0 + (self.steps // 500) * 0.05

    def _get_observation(self):
        # Update camera to follow player smoothly
        target_camera_x = self.player["pos"][0] - self.SCREEN_WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH - self.SCREEN_WIDTH))

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        cam_x = int(self.camera_x)

        # Background decorations
        for dec in self.decorations:
            pygame.draw.rect(self.screen, self.COLOR_DECOR, dec.move(-cam_x, 0))

        # Exit portal glow
        exit_x = self.LEVEL_WIDTH - cam_x
        if exit_x < self.SCREEN_WIDTH + 200:
            glow_radius = int(150 + math.sin(self.steps * 0.1) * 20)
            glow_center = (exit_x, self.LEVEL_GROUND_Y - 100)
            self._draw_glow(glow_center, glow_radius, self.COLOR_EXIT_GLOW, 10)

        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, 0))

        # Spikes
        for spike in self.spikes:
            points = [
                (spike.left - cam_x, spike.bottom),
                (spike.right - cam_x, spike.bottom),
                (spike.centerx - cam_x, spike.top)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SPIKE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SPIKE)

        # Enemies
        for e in self.enemies:
            if e["health"] > 0:
                pos = (int(e["pos"][0] - cam_x), int(e["pos"][1]))
                size = e["size"].astype(int)
                flicker = 1.0 + math.sin(self.steps * 0.5 + e["pos"][0]) * 0.1
                w, h = int(size[0] * flicker), int(size[1] * flicker)
                color = (200, 200, 220)
                pygame.gfxdraw.aaellipse(self.screen, pos[0]+size[0]//2, pos[1]+size[1]//2, w//2, h//2, color)


        # Player
        p = self.player
        p_rect = pygame.Rect(int(p["pos"][0] - cam_x), int(p["pos"][1]), *p["size"])
        
        # Invincibility flash
        if p["invincibility_timer"] > 0 and self.steps % 10 < 5:
            pass # Don't draw player
        else:
            # Body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
            # Eyes
            eye_y = p_rect.y + 10
            eye_x_offset = 5 if p["facing_right"] else 15
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYES, (p_rect.x + eye_x_offset, eye_y), 2)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYES, (p_rect.x + eye_x_offset, eye_y+5), 2)

        # Player Attack
        if p["attack_timer"] > 0:
            x_offset = p_rect.width if p["facing_right"] else -self.PLAYER_ATTACK_RANGE
            attack_world_rect = pygame.Rect(
                p_rect.left + x_offset,
                p_rect.top,
                self.PLAYER_ATTACK_RANGE,
                p_rect.height
            )
            alpha = int(200 * (p["attack_timer"] / self.PLAYER_ATTACK_DURATION))
            self._draw_transparent_rect(attack_world_rect, self.COLOR_ATTACK, alpha)
            

        # Particles
        for part in self.particles:
            pos = (int(part["pos"][0] - cam_x), int(part["pos"][1]))
            alpha = int(255 * (part["life"] / 30.0))
            color = (*part["color"], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, pos)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_glow(self, center, radius, color, alpha_base):
        for i in range(radius, 0, -2):
            alpha = int(alpha_base * (1 - i / radius)**2)
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], i, (*color, alpha))
    
    def _draw_transparent_rect(self, rect, color, alpha):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((*color, alpha))
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, int(bar_width * health_pct), bar_height))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU ESCAPED!" if self.game_won else "YOU DIED"
            text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "player_pos": self.player["pos"].tolist()
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Haunted Mansion Escape")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The environment observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to restart.")

        clock.tick(30) # Match the environment's intended framerate

    env.close()