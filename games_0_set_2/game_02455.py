
# Generated: 2025-08-28T04:59:11.297114
# Source Brief: brief_02455.md
# Brief Index: 2455

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Hold Space to shoot. Survive and reach the glowing green exit!"
    )

    game_description = (
        "Escape hordes of zombies in a side-scrolling survival shooter. Manage your ammo, "
        "clear each stage before time runs out, and fight your way to the final exit."
    )

    auto_advance = True

    # --- CONSTANTS ---
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH_MULTIPLIER = 4
    GROUND_Y = 350
    FPS = 30

    # Player
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40
    PLAYER_SPEED = 5
    PLAYER_JUMP_STRENGTH = 14
    PLAYER_MAX_HEALTH = 30
    PLAYER_START_AMMO = 50
    GRAVITY = 0.7
    SHOT_COOLDOWN = 6  # frames
    INVINCIBILITY_DURATION = 30 # frames

    # Zombie
    ZOMBIE_WIDTH = 20
    ZOMBIE_HEIGHT = 40
    ZOMBIE_HEALTH = 5
    ZOMBIE_BASE_SPEED = 1.0
    MAX_ZOMBIES = 20

    # Projectile
    PROJECTILE_SPEED = 15
    PROJECTILE_WIDTH = 10
    PROJECTILE_HEIGHT = 4

    # Colors
    COLOR_BG_CITY = (20, 25, 40)
    COLOR_BG_ALLEY = (15, 20, 30)
    COLOR_BG_BUILDING = (10, 10, 15)
    COLOR_GROUND = (40, 45, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_EXIT = (50, 255, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_HEALTH_BAR = (0, 255, 0)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_AMMO_BAR = (255, 180, 0)
    COLOR_AMMO_BAR_BG = (80, 80, 80)
    
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_main = pygame.font.Font(None, 48)

        # State variables are initialized in reset()
        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        self.world_width = self.SCREEN_WIDTH * self.WORLD_WIDTH_MULTIPLIER
        self.exit_pos = self.world_width - 100

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.stage = 1
        self.stage_time_limit = 60 * self.FPS
        self.time_left = self.stage_time_limit
        self.max_episode_steps = 1800 # As per brief, though stage logic might end it sooner.

        self.player_pos = pygame.math.Vector2(100, self.GROUND_Y - self.PLAYER_HEIGHT)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_START_AMMO
        self.is_on_ground = False
        self.player_facing_direction = 1
        self.shot_cooldown_timer = 0
        self.player_invincibility_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.camera_x = 0
        self.camera_shake = 0

        self._spawn_initial_zombies(self.MAX_ZOMBIES)
        
        # Parallax background stars
        self.bg_stars = [
            (self.np_random.integers(0, self.world_width), self.np_random.integers(0, self.GROUND_Y), self.np_random.integers(1, 4))
            for _ in range(200)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        self._handle_input(action)
        reward += self._update_player()
        self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        self._update_camera()

        self.steps += 1
        self.time_left -= 1
        if self.shot_cooldown_timer > 0:
            self.shot_cooldown_timer -= 1
        if self.player_invincibility_timer > 0:
            self.player_invincibility_timer -= 1
        
        stage_completion_reward = self._check_stage_completion()
        reward += stage_completion_reward

        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.player_health <= 0 or (self.time_left <= 0 and stage_completion_reward == 0):
                reward -= 100.0  # Death or timeout penalty
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
            self.player_facing_direction = -1
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
            self.player_facing_direction = 1
        else:
            self.player_vel.x = 0

        # Jump
        if movement == 1 and self.is_on_ground:
            self.player_vel.y = -self.PLAYER_JUMP_STRENGTH
            self.is_on_ground = False
            # sfx: jump

        # Shoot
        if space_held and self.shot_cooldown_timer == 0 and self.player_ammo > 0:
            proj_start_pos = self.player_pos + pygame.math.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT / 2)
            self.projectiles.append({
                "pos": proj_start_pos,
                "dir": self.player_facing_direction
            })
            self.player_ammo -= 1
            self.shot_cooldown_timer = self.SHOT_COOLDOWN
            # sfx: shoot

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Move player
        self.player_pos += self.player_vel

        # World boundaries collision
        if self.player_pos.x < 0:
            self.player_pos.x = 0
        if self.player_pos.x > self.world_width - self.PLAYER_WIDTH:
            self.player_pos.x = self.world_width - self.PLAYER_WIDTH

        # Ground collision
        if self.player_pos.y + self.PLAYER_HEIGHT > self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_vel.y = 0
            self.is_on_ground = True
        else:
            self.is_on_ground = False
        
        return -0.01 if self.shot_cooldown_timer == self.SHOT_COOLDOWN -1 else 0 # Ammo usage penalty

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["pos"].x += self.PROJECTILE_SPEED * proj["dir"]
            if not (0 < proj["pos"].x < self.world_width):
                self.projectiles.remove(proj)

    def _update_zombies(self):
        zombie_kill_reward = 0
        zombie_speed = self.ZOMBIE_BASE_SPEED + (self.stage - 1) * 0.5

        for z in self.zombies[:]:
            # Movement
            dir_to_player = 1 if self.player_pos.x > z["pos"].x else -1
            z["pos"].x += dir_to_player * zombie_speed
            
            # Animate legs
            z["anim_timer"] = (z["anim_timer"] + 1) % 20

            # Collision with player
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            zombie_rect = pygame.Rect(z["pos"].x, z["pos"].y, self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
            if player_rect.colliderect(zombie_rect) and self.player_invincibility_timer == 0:
                self.player_health -= 1
                self.player_invincibility_timer = self.INVINCIBILITY_DURATION
                self.camera_shake = 10
                # sfx: player_hit
            
            # Collision with projectiles
            for proj in self.projectiles[:]:
                proj_rect = pygame.Rect(proj["pos"].x, proj["pos"].y, self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT)
                if zombie_rect.colliderect(proj_rect):
                    z["health"] -= 1
                    self.projectiles.remove(proj)
                    # sfx: zombie_hit
                    if z["health"] <= 0:
                        self.zombies.remove(z)
                        self._create_particles(z["pos"] + pygame.math.Vector2(self.ZOMBIE_WIDTH/2, self.ZOMBIE_HEIGHT/2), self.COLOR_ZOMBIE, 15)
                        zombie_kill_reward += 1.0
                        # sfx: zombie_death
                    break
        
        # Respawn zombies
        while len(self.zombies) < self.MAX_ZOMBIES:
            self._spawn_zombie()
        
        return zombie_kill_reward
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.math.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-5, 1)),
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.2 # particle gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.world_width - self.SCREEN_WIDTH))

        if self.camera_shake > 0:
            self.camera_shake -= 1

    def _spawn_zombie(self):
        side = self.np_random.choice([-1, 1])
        spawn_x = -50 if side == -1 else self.world_width + 50
        self.zombies.append({
            "pos": pygame.math.Vector2(spawn_x, self.GROUND_Y - self.ZOMBIE_HEIGHT),
            "health": self.ZOMBIE_HEALTH,
            "anim_timer": self.np_random.integers(0, 20)
        })

    def _spawn_initial_zombies(self, count):
        for _ in range(count):
            spawn_x = self.np_random.integers(0, self.world_width)
            # Avoid spawning on player
            if abs(spawn_x - self.player_pos.x) < self.SCREEN_WIDTH / 2:
                spawn_x += self.SCREEN_WIDTH * self.np_random.choice([-1, 1])
            
            self.zombies.append({
                "pos": pygame.math.Vector2(spawn_x, self.GROUND_Y - self.ZOMBIE_HEIGHT),
                "health": self.ZOMBIE_HEALTH,
                "anim_timer": self.np_random.integers(0, 20)
            })

    def _check_stage_completion(self):
        if self.player_pos.x + self.PLAYER_WIDTH >= self.exit_pos:
            self.stage += 1
            if self.stage > 3:
                self.game_over = True # Game won!
                return 100.0
            else:
                # Reset for next stage
                self.player_pos.x = 100
                self.time_left = self.stage_time_limit
                self.zombies.clear()
                self.projectiles.clear()
                self.particles.clear()
                self._spawn_initial_zombies(self.MAX_ZOMBIES)
                return 100.0
        return 0.0

    def _check_termination(self):
        return self.player_health <= 0 or self.time_left <= 0 or self.game_over or self.steps >= self.max_episode_steps

    def _get_observation(self):
        # Camera offset for shake
        cam_offset_x = 0
        cam_offset_y = 0
        if self.camera_shake > 0:
            cam_offset_x = self.np_random.integers(-5, 6)
            cam_offset_y = self.np_random.integers(-5, 6)
        
        final_camera_x = self.camera_x + cam_offset_x

        # --- Rendering ---
        self._render_background(final_camera_x)
        self._render_ground(final_camera_x)
        self._render_exit(final_camera_x)
        self._render_zombies(final_camera_x)
        self._render_player(final_camera_x)
        self._render_projectiles(final_camera_x)
        self._render_particles(final_camera_x)
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_x):
        bg_color = self.COLOR_BG_CITY
        if self.stage == 2: bg_color = self.COLOR_BG_ALLEY
        elif self.stage >= 3: bg_color = self.COLOR_BG_BUILDING
        self.screen.fill(bg_color)
        
        # Parallax stars/lights
        for x, y, speed in self.bg_stars:
            screen_x = (x - cam_x / speed) % self.SCREEN_WIDTH
            pygame.draw.rect(self.screen, (100, 100, 120), (screen_x, y, speed, speed))

    def _render_ground(self, cam_x):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_exit(self, cam_x):
        exit_rect = pygame.Rect(self.exit_pos - cam_x, self.GROUND_Y - 100, 20, 100)
        # Glow effect
        for i in range(10, 0, -1):
            alpha = 150 - i * 15
            color = (self.COLOR_EXIT[0], self.COLOR_EXIT[1], self.COLOR_EXIT[2], alpha)
            s = pygame.Surface((exit_rect.width + i*2, exit_rect.height + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=5)
            self.screen.blit(s, (exit_rect.x - i, exit_rect.y - i))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=3)
        
    def _render_zombies(self, cam_x):
        for z in self.zombies:
            z_rect = pygame.Rect(z["pos"].x - cam_x, z["pos"].y, self.ZOMBIE_WIDTH, self.ZOMBIE_HEIGHT)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)
            # Leg animation
            leg_offset = 10 if z["anim_timer"] < 10 else -10
            pygame.draw.line(self.screen, self.COLOR_ZOMBIE, z_rect.midbottom, (z_rect.centerx + leg_offset, z_rect.bottom + 10), 3)

    def _render_player(self, cam_x):
        # Bobbing animation for running
        bob = 0
        if self.is_on_ground and self.player_vel.x != 0:
            bob = abs(math.sin(self.steps * 0.5)) * 3

        player_rect = pygame.Rect(self.player_pos.x - cam_x, self.player_pos.y - bob, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        color = self.COLOR_PLAYER
        if self.player_invincibility_timer > 0 and self.steps % 4 < 2:
            color = (255, 255, 255) # Flash white when invincible

        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)
        
        # Gun
        gun_y = player_rect.centery
        if self.player_facing_direction == 1:
            gun_rect = pygame.Rect(player_rect.right, gun_y - 2, 10, 4)
        else:
            gun_rect = pygame.Rect(player_rect.left - 10, gun_y - 2, 10, 4)
        pygame.draw.rect(self.screen, color, gun_rect)

    def _render_projectiles(self, cam_x):
        for proj in self.projectiles:
            proj_rect = pygame.Rect(proj["pos"].x - cam_x, proj["pos"].y, self.PROJECTILE_WIDTH, self.PROJECTILE_HEIGHT)
            pygame.gfxdraw.box(self.screen, proj_rect, self.COLOR_PROJECTILE)
            pygame.gfxdraw.box(self.screen, proj_rect.inflate(4,4), (*self.COLOR_PROJECTILE, 50))

    def _render_particles(self, cam_x):
        for p in self.particles:
            alpha = max(0, min(255, p["life"] * 15))
            color = (*p["color"], alpha)
            size = max(1, int(p["life"] / 6))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (p["pos"].x - cam_x - size, p["pos"].y - size))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, 200 * health_ratio, 20))

        # Ammo Counter
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (self.SCREEN_WIDTH - ammo_text.get_width() - 10, 10))

        # Stage and Time
        time_sec = self.time_left // self.FPS
        stage_text = self.font_ui.render(f"STAGE: {self.stage} | TIME: {time_sec}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH/2 - stage_text.get_width()/2, self.SCREEN_HEIGHT - 30))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "MISSION COMPLETE" if self.stage > 3 else "YOU DIED"
        text = self.font_main.render(message, True, (255, 50, 50) if message == "YOU DIED" else (50, 255, 50))
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "time_left": self.time_left // self.FPS
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for manual play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Escape")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # --- Game Loop ---
    while not terminated:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Action Mapping for Keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to Screen ---
        # The observation is already a rendered image, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Print Info ---
        if env.steps % GameEnv.FPS == 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Health: {info['health']}, Ammo: {info['ammo']}, Stage: {info['stage']}")

        # --- Cap Framerate ---
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    
    # Keep the window open for a bit to see the game over screen
    pygame.time.wait(2000)
    
    env.close()