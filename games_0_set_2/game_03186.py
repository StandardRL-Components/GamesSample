
# Generated: 2025-08-27T22:37:42.464173
# Source Brief: brief_03186.md
# Brief Index: 3186

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press Shift to rotate aim. Press Space to shoot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in an isometric arena for 60 seconds. "
        "Move to avoid contact, aim, and shoot to clear a path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.W, self.H = 640, 400
        self.ARENA_W, self.ARENA_H = 550, 350
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.C_BG = (34, 34, 40)
        self.C_ARENA = (50, 50, 55)
        self.C_PLAYER = (0, 255, 150)
        self.C_PLAYER_GLOW = (0, 255, 150, 50)
        self.C_ZOMBIE = (255, 50, 50)
        self.C_ZOMBIE_GLOW = (255, 50, 50, 40)
        self.C_BLOOD = (200, 0, 0)
        self.C_MUZZLE_FLASH = (255, 255, 0)
        self.C_HEALTH_GOOD = (0, 200, 100)
        self.C_HEALTH_BAD = (150, 0, 0)
        self.C_WHITE = (255, 255, 255)

        # Game constants
        self.MAX_STEPS = 1800  # 60 seconds at 30fps
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 3.0
        self.PLAYER_SIZE = 10
        self.ZOMBIE_MAX_HEALTH = 40
        self.ZOMBIE_SPEED = 0.75
        self.ZOMBIE_SIZE = 8
        self.ZOMBIE_DAMAGE = 10
        self.BULLET_DAMAGE = 20
        self.NUM_ZOMBIES = 50
        self.SHOOT_COOLDOWN = 6  # frames
        
        self.arena_rect = pygame.Rect(
            (self.W - self.ARENA_W) // 2,
            (self.H - self.ARENA_H) // 2,
            self.ARENA_W,
            self.ARENA_H
        )

        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.kill_count = 0
        self.game_over = False
        self.victory = False

        self.player_pos = pygame.Vector2(self.W / 2, self.H / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = 0.0
        
        self.shoot_cooldown_timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.zombies = []
        for _ in range(self.NUM_ZOMBIES):
            self._spawn_zombie()

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _spawn_zombie(self):
        # Spawn zombies outside a radius around the player
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(self.arena_rect.left, self.arena_rect.right),
                self.np_random.uniform(self.arena_rect.top, self.arena_rect.bottom)
            )
            if pos.distance_to(self.player_pos) > 150:
                self.zombies.append({
                    "pos": pos,
                    "health": self.ZOMBIE_MAX_HEALTH
                })
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.01  # Survival reward per frame

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_player()
        zombie_damage, kills = self._update_zombies()
        self.player_health -= zombie_damage
        reward += kills * 1.0
        self.kill_count += kills

        self._update_particles()
        
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0:
            self.player_health = 0
            terminated = True
            self.game_over = True
            self.victory = False
            reward = -100.0  # Large penalty for dying
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.victory = True
            reward += 50.0  # Large reward for winning

        self.score = self.kill_count # Score is just the kill count
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        self.player_vel = pygame.Vector2(0, 0)
        if movement == 1: self.player_vel.y = -1 # Up
        elif movement == 2: self.player_vel.y = 1  # Down
        elif movement == 3: self.player_vel.x = -1 # Left
        elif movement == 4: self.player_vel.x = 1  # Right
        
        if self.player_vel.length() > 0:
            self.player_vel.scale_to_length(self.PLAYER_SPEED)

        # Aiming (rotate on key press)
        if shift_held and not self.prev_shift_held:
            self.player_aim_angle += math.pi / 4  # 45 degrees
            if self.player_aim_angle > 2 * math.pi:
                self.player_aim_angle -= 2 * math.pi
        
        # Shooting (shoot on key press)
        if space_held and not self.prev_space_held and self.shoot_cooldown_timer == 0:
            self._shoot()
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, self.arena_rect.left + self.PLAYER_SIZE, self.arena_rect.right - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.arena_rect.top + self.PLAYER_SIZE, self.arena_rect.bottom - self.PLAYER_SIZE)

    def _update_zombies(self):
        total_damage = 0
        kills = 0
        for zombie in reversed(self.zombies):
            # Movement
            direction = (self.player_pos - zombie["pos"]).normalize()
            zombie["pos"] += direction * self.ZOMBIE_SPEED
            
            # Player collision
            if zombie["pos"].distance_to(self.player_pos) < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                total_damage += self.ZOMBIE_DAMAGE
                # Knockback
                zombie["pos"] -= direction * (self.PLAYER_SIZE + self.ZOMBIE_SIZE) * 1.1
                # SFX: Player hurt
        return total_damage, kills

    def _shoot(self):
        self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        # SFX: Gunshot
        
        # Muzzle Flash
        flash_pos = self.player_pos + pygame.Vector2(self.PLAYER_SIZE, 0).rotate_rad(self.player_aim_angle)
        self._create_particles(flash_pos, 1, self.C_MUZZLE_FLASH, 3, 15, 'star')

        # Hitscan logic
        aim_vec = pygame.Vector2(1, 0).rotate_rad(self.player_aim_angle)
        
        closest_hit = None
        min_dist = float('inf')

        for zombie in self.zombies:
            player_to_zombie = zombie["pos"] - self.player_pos
            dist_along_ray = player_to_zombie.dot(aim_vec)

            if dist_along_ray > 0:
                perp_dist = (player_to_zombie - dist_along_ray * aim_vec).length()
                if perp_dist < self.ZOMBIE_SIZE:
                    if dist_along_ray < min_dist:
                        min_dist = dist_along_ray
                        closest_hit = zombie
        
        # Tracer line
        end_pos = self.player_pos + aim_vec * self.W
        self._create_particles(self.player_pos, 1, self.C_WHITE, 2, 0, 'line', end_point=end_pos)

        if closest_hit:
            closest_hit["health"] -= self.BULLET_DAMAGE
            # SFX: Flesh hit
            self._create_particles(closest_hit["pos"], 15, self.C_BLOOD, 10, 3)
            
            if closest_hit["health"] <= 0:
                # SFX: Zombie death
                self._create_particles(closest_hit["pos"], 30, self.C_BLOOD, 20, 5)
                self.zombies.remove(closest_hit)
                self.kill_count += 1
                self.score = self.kill_count

    def _create_particles(self, pos, count, color, life, speed, p_type='circle', end_point=None):
        for _ in range(count):
            if p_type == 'line':
                 self.particles.append({"pos": pos.copy(), "end_pos": end_point, "life": life, "max_life": life, "color": color, "type": p_type})
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(0.5, speed)
                self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "max_life": life, "color": color, "type": p_type})

    def _update_particles(self):
        for p in reversed(self.particles):
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            elif p['type'] != 'line':
                p["pos"] += p["vel"]
                p["vel"] *= 0.95 # friction

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.C_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena
        pygame.draw.rect(self.screen, self.C_ARENA, self.arena_rect)

        # Draw shadows
        for z in self.zombies:
            pygame.gfxdraw.filled_ellipse(self.screen, int(z["pos"].x), int(z["pos"].y + self.ZOMBIE_SIZE/2), self.ZOMBIE_SIZE, self.ZOMBIE_SIZE // 2, (0,0,0,80))
        pygame.gfxdraw.filled_ellipse(self.screen, int(self.player_pos.x), int(self.player_pos.y + self.PLAYER_SIZE/2), self.PLAYER_SIZE, self.PLAYER_SIZE // 2, (0,0,0,100))

        # Draw zombies
        for z in self.zombies:
            pygame.gfxdraw.filled_circle(self.screen, int(z["pos"].x), int(z["pos"].y), self.ZOMBIE_SIZE, self.C_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, int(z["pos"].x), int(z["pos"].y), self.ZOMBIE_SIZE, self.C_ZOMBIE)

        # Draw player
        p_x, p_y = int(self.player_pos.x), int(self.player_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, p_x, p_y, self.PLAYER_SIZE, self.C_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, p_x, p_y, self.PLAYER_SIZE, self.C_PLAYER)
        
        # Draw aiming line
        aim_vec = pygame.Vector2(self.PLAYER_SIZE * 1.5, 0).rotate_rad(self.player_aim_angle)
        end_pos = self.player_pos + aim_vec
        pygame.draw.line(self.screen, self.C_WHITE, self.player_pos, end_pos, 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            if p['type'] == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 2, p["color"] + (alpha,))
            elif p['type'] == 'star':
                size = int(p['life'] * 3)
                pygame.draw.line(self.screen, p['color'], (p['pos'].x - size, p['pos'].y), (p['pos'].x + size, p['pos'].y), 3)
                pygame.draw.line(self.screen, p['color'], (p['pos'].x, p['pos'].y - size), (p['pos'].x, p['pos'].y + size), 3)
            elif p['type'] == 'line':
                pygame.draw.aaline(self.screen, p['color'], p['pos'], p['end_pos'], 1)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.C_HEALTH_BAD, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.C_HEALTH_GOOD, (10, 10, int(bar_width * health_ratio), bar_height))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        timer_text = f"{int(time_left // 60):02}:{int(time_left % 60):02}"
        text_surf = self.font_ui.render(timer_text, True, self.C_WHITE)
        self.screen.blit(text_surf, (self.W - text_surf.get_width() - 10, 10))

        # Kill Count
        kill_text = f"KILLS: {self.kill_count}"
        text_surf = self.font_ui.render(kill_text, True, self.C_WHITE)
        self.screen.blit(text_surf, ((self.W - text_surf.get_width()) // 2, self.H - text_surf.get_height() - 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.victory else "GAME OVER"
            text_surf = self.font_big.render(msg, True, self.C_WHITE)
            self.screen.blit(text_surf, ((self.W - text_surf.get_width()) / 2, (self.H - text_surf.get_height()) / 2 - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "kills": self.kill_count,
            "health": self.player_health,
            "time_left": (self.MAX_STEPS - self.steps) / 30,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a display for human play
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        if terminated:
            print(f"Game Over! Final Score (Kills): {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            env.reset()

        clock.tick(30) # 30 FPS

    env.close()