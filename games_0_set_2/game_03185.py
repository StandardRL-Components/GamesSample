import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Pygame must run headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: ←→ to run, ↑ to jump. Press space to attack."

    # Short, user-facing description of the game
    game_description = "A fast-paced platformer where a ninja navigates obstacle courses and defeats enemies."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PLATFORM = (80, 80, 90)
    COLOR_PLAYER = (0, 255, 127) # Spring Green
    COLOR_ENEMY = (255, 69, 0) # OrangeRed
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (0, 200, 83)
    COLOR_HEALTH_BAR_BG = (60, 60, 60)
    COLOR_ATTACK = (255, 255, 255)

    # Physics
    GRAVITY = 0.6
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.15
    PLAYER_JUMP_STRENGTH = -12
    MAX_VEL_X = 6
    MAX_VEL_Y = 15

    # Game Parameters
    MAX_STEPS = 1500
    PLAYER_START_HEALTH = 100
    LEVEL_WIDTH_FACTOR = 5 
    NUM_ENEMIES = 10
    
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.platforms = []
        self.particles = []
        self.camera_offset = pygame.Vector2(0, 0)
        self.level_end_x = 0
        self.rng = None

        self.reset()
        # The original code called validate_implementation, but it's not part of the standard API
        # and can be removed. However, to match the execution that caused the error, we keep it.
        # self.validate_implementation() 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player = {
            "pos": pygame.Vector2(100, 200),
            "vel": pygame.Vector2(0, 0),
            "rect": pygame.Rect(0, 0, 20, 30),
            "health": self.PLAYER_START_HEALTH,
            "on_ground": False,
            "is_attacking": False,
            "attack_timer": 0,
            "attack_cooldown": 20, # frames
            "invincible_timer": 0,
            "facing_right": True
        }

        self._generate_level()
        self._spawn_enemies()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_input(action)
        reward += self._update_game_state(action)
        reward += self._handle_collisions()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            if self.player["health"] > 0 and self.player["pos"].x >= self.level_end_x: # Reached end
                reward += 100
                self.score += 1000
            self.game_over = True
        
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        self.platforms = []
        level_width = self.SCREEN_WIDTH * self.LEVEL_WIDTH_FACTOR
        
        # Start platform
        start_platform = pygame.Rect(0, 350, 300, 50)
        self.platforms.append(start_platform)
        self.player["pos"] = pygame.Vector2(100, start_platform.top - self.player["rect"].height)

        current_x = start_platform.width
        last_y = start_platform.y

        while current_x < level_width:
            gap = self.rng.integers(40, 120)
            width = self.rng.integers(100, 350)
            y_change = self.rng.integers(-90, 90)
            
            new_x = current_x + gap
            new_y = np.clip(last_y + y_change, 150, 380)
            
            platform_rect = pygame.Rect(new_x, new_y, width, 20)
            self.platforms.append(platform_rect)
            
            current_x = new_x + width
            last_y = new_y
        
        self.level_end_x = current_x - 100

    def _spawn_enemies(self):
        self.enemies = []
        valid_platforms = self.platforms[1:]
        if not valid_platforms:
            return

        for _ in range(self.NUM_ENEMIES):
            # FIX: np.random.choice converts the list of Rects into a 2D numpy array
            # and returns a row (a 1D numpy array), which doesn't have .x or .width attributes.
            # The fix is to select an index and then get the object from the list.
            platform_index = self.rng.integers(0, len(valid_platforms))
            platform = valid_platforms[platform_index]
            
            enemy = {
                "pos": pygame.Vector2(platform.x + platform.width / 2, platform.y - 25),
                "vel": pygame.Vector2(self.rng.choice([-1, 1]), 0),
                "rect": pygame.Rect(0, 0, 25, 25),
                "patrol_start": platform.x,
                "patrol_end": platform.right,
                "is_alive": True,
                "respawn_timer": 0,
                "respawn_duration": 300 # 5 seconds at 60fps
            }
            self.enemies.append(enemy)

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if movement == 3:  # Left
            self.player["vel"].x -= self.PLAYER_ACCEL
            self.player["facing_right"] = False
        elif movement == 4:  # Right
            self.player["vel"].x += self.PLAYER_ACCEL
            self.player["facing_right"] = True
        
        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"].y = self.PLAYER_JUMP_STRENGTH
            self.player["on_ground"] = False
            self._create_particles(self.player["pos"] + pygame.Vector2(self.player["rect"].width/2, self.player["rect"].height), 5, self.COLOR_PLATFORM)


        # Attacking
        if space_held and self.player["attack_timer"] == 0:
            self.player["is_attacking"] = True
            self.player["attack_timer"] = self.player["attack_cooldown"]

    def _update_game_state(self, action):
        movement, _, _ = action
        # --- Update Player ---
        # Apply friction
        if self.player["vel"].x != 0:
            self.player["vel"].x += self.player["vel"].x * self.PLAYER_FRICTION
            if abs(self.player["vel"].x) < 0.1: self.player["vel"].x = 0
        
        # Apply gravity
        self.player["vel"].y += self.GRAVITY
        
        # Clamp velocities
        self.player["vel"].x = np.clip(self.player["vel"].x, -self.MAX_VEL_X, self.MAX_VEL_X)
        self.player["vel"].y = np.clip(self.player["vel"].y, -self.MAX_VEL_Y, self.MAX_VEL_Y)

        # Update position (handle collisions separately)
        self.player["pos"].x += self.player["vel"].x
        self.player["pos"].y += self.player["vel"].y
        
        self.player["rect"].topleft = self.player["pos"]
        
        # Assume not on ground until collision check proves otherwise
        self.player["on_ground"] = False

        # Update timers
        if self.player["attack_timer"] > 0:
            self.player["attack_timer"] -= 1
            if self.player["attack_timer"] == 0:
                self.player["is_attacking"] = False
        
        if self.player["invincible_timer"] > 0:
            self.player["invincible_timer"] -= 1

        # --- Update Enemies ---
        for enemy in self.enemies:
            if enemy["is_alive"]:
                enemy["pos"] += enemy["vel"]
                if enemy["pos"].x <= enemy["patrol_start"] or enemy["pos"].x >= enemy["patrol_end"] - enemy["rect"].width:
                    enemy["vel"].x *= -1
                enemy["rect"].topleft = enemy["pos"]
            else:
                if enemy["respawn_timer"] > 0:
                    enemy["respawn_timer"] -= 1
                else:
                    enemy["is_alive"] = True

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # --- Calculate Reward ---
        reward = 0
        if movement == 4: reward += 0.01 # Reward for moving right
        if movement == 3: reward -= 0.005 # Small penalty for moving left
        return reward

    def _handle_collisions(self):
        collision_reward = 0
        # --- Player vs Platforms ---
        for plat in self.platforms:
            if self.player["rect"].colliderect(plat):
                # Vertical collision
                if self.player["vel"].y > 0 and self.player["rect"].bottom > plat.top and self.player["rect"].bottom < plat.top + self.player["vel"].y + 1:
                    self.player["pos"].y = plat.top - self.player["rect"].height
                    self.player["vel"].y = 0
                    if not self.player["on_ground"]: # Landing
                        self._create_particles(self.player["pos"] + pygame.Vector2(self.player["rect"].width/2, self.player["rect"].height), 5, self.COLOR_PLATFORM)
                    self.player["on_ground"] = True
                # Horizontal collision
                elif self.player["vel"].x > 0 and self.player["rect"].right > plat.left and self.player["rect"].right < plat.left + self.player["vel"].x + 1:
                    self.player["pos"].x = plat.left - self.player["rect"].width
                    self.player["vel"].x = 0
                elif self.player["vel"].x < 0 and self.player["rect"].left < plat.right and self.player["rect"].left > plat.right + self.player["vel"].x - 1:
                    self.player["pos"].x = plat.right
                    self.player["vel"].x = 0
                # Ceiling collision
                elif self.player["vel"].y < 0 and self.player["rect"].top < plat.bottom and self.player["rect"].top > plat.bottom + self.player["vel"].y - 1:
                     self.player["pos"].y = plat.bottom
                     self.player["vel"].y = 0

        self.player["rect"].topleft = self.player["pos"]
        
        # --- Attack vs Enemies ---
        if self.player["is_attacking"] and self.player["attack_timer"] > self.player["attack_cooldown"] - 5: # Active attack frames
            attack_offset = 25 if self.player["facing_right"] else -45
            attack_rect = pygame.Rect(self.player["rect"].centerx + attack_offset, self.player["rect"].centery - 20, 40, 40)
            
            for enemy in self.enemies:
                if enemy["is_alive"] and attack_rect.colliderect(enemy["rect"]):
                    enemy["is_alive"] = False
                    enemy["respawn_timer"] = enemy["respawn_duration"]
                    self.score += 50
                    collision_reward += 10 # FIX: Correctly add reward
                    self._create_particles(enemy["pos"] + pygame.Vector2(enemy["rect"].width/2, enemy["rect"].height/2), 15, self.COLOR_ENEMY)

        # --- Player vs Enemies ---
        if self.player["invincible_timer"] == 0:
            for enemy in self.enemies:
                if enemy["is_alive"] and self.player["rect"].colliderect(enemy["rect"]):
                    self.player["health"] -= 10
                    self.player["invincible_timer"] = 60 # 1 sec invincibility
                    collision_reward -= 0.1 # FIX: Correctly add penalty
                    self._create_particles(self.player["pos"] + self.player["rect"].center, 20, self.COLOR_PLAYER)
                    # Knockback
                    knockback_dir = 1 if self.player["pos"].x < enemy["pos"].x else -1
                    self.player["vel"].x = -knockback_dir * 5
                    self.player["vel"].y = -4
                    break
        return collision_reward
    
    def _check_termination(self):
        if self.player["health"] <= 0:
            self.player["health"] = 0
            return True
        if self.player["pos"].x >= self.level_end_x:
            return True
        if self.player["pos"].y > self.SCREEN_HEIGHT + 100: # Fell off world
            self.player["health"] = 0
            return True
        return False

    def _get_observation(self):
        self._update_camera()
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_platforms()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_camera(self):
        # Center camera on player with a lookahead, clamped to level bounds
        target_x = -self.player["pos"].x + self.SCREEN_WIDTH / 2 - 100
        target_y = -self.player["pos"].y + self.SCREEN_HEIGHT / 2 + 50
        
        # Smooth camera movement
        self.camera_offset.x += (target_x - self.camera_offset.x) * 0.1
        self.camera_offset.y += (target_y - self.camera_offset.y) * 0.1

        # Clamp camera
        self.camera_offset.x = min(0, self.camera_offset.x)
        self.camera_offset.x = max(-(self.level_end_x + 100 - self.SCREEN_WIDTH), self.camera_offset.x)
        self.camera_offset.y = min(100, self.camera_offset.y)
        self.camera_offset.y = max(-200, self.camera_offset.y)


    def _render_background(self):
        # Draw a parallax grid
        for i in range(0, self.SCREEN_WIDTH, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))
        # End of level marker
        end_line_x = int(self.level_end_x + self.camera_offset.x)
        if 0 < end_line_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (end_line_x, 0), (end_line_x, self.SCREEN_HEIGHT), 3)


    def _render_platforms(self):
        for plat in self.platforms:
            # Cull platforms not on screen
            if plat.right + self.camera_offset.x > 0 and plat.left + self.camera_offset.x < self.SCREEN_WIDTH:
                render_rect = plat.move(self.camera_offset)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect, border_radius=3)

    def _render_player(self):
        render_rect = self.player["rect"].move(self.camera_offset)
        
        # Invincibility flash
        if self.player["invincible_timer"] > 0 and (self.steps // 3) % 2 == 0:
            return

        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, render_rect, border_radius=4)
        
        # Eyes
        eye_y = render_rect.top + 8
        if self.player["facing_right"]:
            eye_x1 = render_rect.right - 8
            eye_x2 = render_rect.right - 4
        else:
            eye_x1 = render_rect.left + 4
            eye_x2 = render_rect.left + 8
        pygame.draw.circle(self.screen, self.COLOR_BG, (eye_x1, eye_y), 2)
        pygame.draw.circle(self.screen, self.COLOR_BG, (eye_x2, eye_y), 2)

        # Attack swoosh
        if self.player["is_attacking"]:
            progress = 1 - (self.player["attack_timer"] / self.player["attack_cooldown"])
            angle_start = math.pi * 1.25 if self.player["facing_right"] else math.pi * 0.25
            angle_end = math.pi * 1.75 if self.player["facing_right"] else math.pi * 0.75
            
            center = render_rect.center
            radius = 30
            
            swoosh_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius*2, radius*2)
            
            alpha = int(200 * (1 - progress))
            if alpha > 0:
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.arc(s, (*self.COLOR_ATTACK, alpha), (0,0,radius*2,radius*2), angle_start, angle_end, width=5)
                self.screen.blit(s, swoosh_rect.topleft)

    def _render_enemies(self):
        for enemy in self.enemies:
            if enemy["is_alive"]:
                render_rect = enemy["rect"].move(self.camera_offset)
                if render_rect.right > 0 and render_rect.left < self.SCREEN_WIDTH:
                    pygame.draw.rect(self.screen, self.COLOR_ENEMY, render_rect, border_radius=5)
                    # "Eye" to show direction
                    eye_pos_x = render_rect.centerx + enemy["vel"].x * 5
                    eye_pos_y = render_rect.centery - 5
                    pygame.draw.circle(self.screen, self.COLOR_BG, (eye_pos_x, eye_pos_y), 3)

    def _render_particles(self):
        for p in self.particles:
            pos = p["pos"] + self.camera_offset
            # Cull particles
            if 0 < pos.x < self.SCREEN_WIDTH and 0 < pos.y < self.SCREEN_HEIGHT:
                radius = int(p["life"] / p["start_life"] * p["size"])
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, p["color"])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Health Bar
        health_pct = self.player["health"] / self.PLAYER_START_HEALTH
        bar_width = 200
        bar_height = 20
        
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=4)
        
        fill_width = max(0, bar_width * health_pct)
        fill_rect = pygame.Rect(10, 10, fill_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect, border_radius=4)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            life = self.rng.integers(15, 30)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "life": life,
                "start_life": life,
                "size": self.rng.integers(3, 6),
                "color": color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "player_pos": (self.player["pos"].x, self.player["pos"].y),
        }

    def close(self):
        pygame.quit()