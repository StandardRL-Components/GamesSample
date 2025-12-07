import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A Gymnasium environment for a top-down zombie survival arcade shooter.

    The player must survive 5 waves of zombies. Each wave is progressively harder,
    with more zombies that move faster. The player can move in four directions
    and shoot projectiles to defeat the zombies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift to aim clockwise. Press Space to fire."
    )

    # Short, user-facing description of the game
    game_description = (
        "Survive waves of zombies in a top-down arcade shooter. "
        "Strategically move and shoot to clear 5 waves and win."
    )

    # Frames auto-advance at a fixed rate for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.W, self.H = 640, 400

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (50, 200, 100)
        self.COLOR_PLAYER_FLASH = (200, 255, 200)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_ZOMBIE_FLASH = (255, 150, 150)
        self.COLOR_BULLET = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        self.COLOR_MSG = (255, 255, 100)
        
        # Fonts
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 50)

        # Player settings
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 4
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_INVINCIBILITY_FRAMES = 60

        # Zombie settings
        self.ZOMBIE_SIZE = 16
        self.ZOMBIE_BASE_COUNT = 20
        self.ZOMBIE_COUNT_INCREASE = 5
        self.ZOMBIE_BASE_SPEED = 0.8
        self.ZOMBIE_SPEED_INCREASE = 0.1
        self.ZOMBIE_AI_UPDATE_RATE = 10

        # Bullet settings
        self.BULLET_SIZE = 4
        self.BULLET_SPEED = 8
        self.BULLET_COOLDOWN = 8 # frames

        # Game rules
        self.MAX_WAVES = 5
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps

        # Player direction vectors (Up, Right, Down, Left)
        self.DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.player_direction_idx = 0
        self.player_hit_timer = 0
        self.bullet_cooldown_timer = 0
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.wave = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.game_message = ""
        self.message_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.W / 2, self.H / 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_direction_idx = 0  # Start facing up
        self.player_hit_timer = 0
        self.bullet_cooldown_timer = 0

        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.wave = 1
        self._spawn_wave()

        self.last_space_held = False
        self.last_shift_held = False

        self.game_message = f"WAVE {self.wave}"
        self.message_timer = 60

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Survival reward
        reward += 0.01

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- Update Game Logic ---
        self._update_player(movement, space_pressed, shift_pressed)
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        # --- Handle Collisions & Rewards ---
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        # --- Check Game State Progression ---
        wave_rewards = self._check_wave_completion()
        reward += wave_rewards

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            self.win = False
            reward -= 10.0  # Loss penalty
            self.game_message = "GAME OVER"
            self.message_timer = 120
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 50)
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            reward -= 10.0  # Time-up is a loss
            self.game_message = "TIME UP!"
            self.message_timer = 120

        terminated = self.game_over
        self.steps += 1
        
        if self.message_timer > 0:
            self.message_timer -= 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, space_pressed, shift_pressed):
        # Movement
        if movement > 0:
            move_map = {1: 0, 2: 2, 3: 3, 4: 1} # Action to direction index
            direction_idx = move_map[movement]
            self.player_direction_idx = direction_idx
            dx, dy = self.DIRECTIONS[direction_idx]
            self.player_pos[0] += dx * self.PLAYER_SPEED
            self.player_pos[1] += dy * self.PLAYER_SPEED

        # Aiming
        if shift_pressed:
            self.player_direction_idx = (self.player_direction_idx + 1) % 4
            # sfx: aim_click.wav

        # Shooting
        if self.bullet_cooldown_timer > 0:
            self.bullet_cooldown_timer -= 1
            
        if space_pressed and self.bullet_cooldown_timer == 0:
            self.bullet_cooldown_timer = self.BULLET_COOLDOWN
            direction = self.DIRECTIONS[self.player_direction_idx]
            # Spawn bullet slightly in front of the player
            start_pos = [
                self.player_pos[0] + direction[0] * self.PLAYER_SIZE,
                self.player_pos[1] + direction[1] * self.PLAYER_SIZE
            ]
            self.bullets.append({
                "pos": start_pos,
                "vel": [direction[0] * self.BULLET_SPEED, direction[1] * self.BULLET_SPEED]
            })
            # sfx: shoot.wav
        
        # Keep player in bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.W - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.H - self.PLAYER_SIZE / 2)

        # Update hit timer
        if self.player_hit_timer > 0:
            self.player_hit_timer -= 1

    def _update_bullets(self):
        for bullet in self.bullets:
            bullet["pos"][0] += bullet["vel"][0]
            bullet["pos"][1] += bullet["vel"][1]
        
        # Remove bullets that are off-screen
        self.bullets = [b for b in self.bullets if 0 < b["pos"][0] < self.W and 0 < b["pos"][1] < self.H]

    def _update_zombies(self):
        zombie_speed = self.ZOMBIE_BASE_SPEED + (self.wave - 1) * self.ZOMBIE_SPEED_INCREASE
        for zombie in self.zombies:
            # AI update logic
            zombie["ai_timer"] -= 1
            if zombie["ai_timer"] <= 0:
                zombie["ai_timer"] = self.ZOMBIE_AI_UPDATE_RATE
                
                # 80% chance to move towards player, 20% to move randomly
                if self.np_random.random() < 0.8:
                    angle_to_player = math.atan2(self.player_pos[1] - zombie["pos"][1], self.player_pos[0] - zombie["pos"][0])
                    zombie["vel"] = [math.cos(angle_to_player), math.sin(angle_to_player)]
                else:
                    rand_dir = self.np_random.integers(0, 4)
                    zombie["vel"] = self.DIRECTIONS[rand_dir]

            # Move zombie
            zombie["pos"][0] += zombie["vel"][0] * zombie_speed
            zombie["pos"][1] += zombie["vel"][1] * zombie_speed
            
            # Bobbing animation
            zombie["anim_timer"] = (zombie["anim_timer"] + 1) % 20
    
    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions(self):
        reward = 0.0
        
        # Bullets vs Zombies
        zombies_to_keep = []
        zombies_hit_indices = set()
        
        for bullet in self.bullets[:]:
            hit = False
            for i, zombie in enumerate(self.zombies):
                if i in zombies_hit_indices: continue
                dist = math.hypot(bullet["pos"][0] - zombie["pos"][0], bullet["pos"][1] - zombie["pos"][1])
                if dist < (self.BULLET_SIZE / 2 + self.ZOMBIE_SIZE / 2):
                    self.bullets.remove(bullet)
                    zombies_hit_indices.add(i)
                    self.score += 10
                    reward += 1.0 # Kill reward
                    self._create_particles(zombie["pos"], self.COLOR_ZOMBIE, 20)
                    # sfx: zombie_death.wav
                    hit = True
                    break
            if hit: continue
        
        self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_hit_indices]

        # Player vs Zombies
        if self.player_hit_timer == 0:
            for zombie in self.zombies:
                dist = math.hypot(self.player_pos[0] - zombie["pos"][0], self.player_pos[1] - zombie["pos"][1])
                if dist < (self.PLAYER_SIZE / 2 + self.ZOMBIE_SIZE / 2):
                    self.player_health -= 25
                    self.player_health = max(0, self.player_health)
                    self.player_hit_timer = self.PLAYER_INVINCIBILITY_FRAMES
                    reward -= 0.5 # Damage penalty
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 15)
                    # sfx: player_hit.wav
                    break # Only take damage from one zombie per frame
        return reward

    def _check_wave_completion(self):
        if not self.zombies and not self.game_over:
            if self.wave >= self.MAX_WAVES:
                self.game_over = True
                self.win = True
                self.score += 500
                self.game_message = "YOU WIN!"
                self.message_timer = 120
                # sfx: win_jingle.wav
                return 50.0 # Win reward
            else:
                self.wave += 1
                self.score += 100
                self._spawn_wave()
                self.game_message = f"WAVE {self.wave}"
                self.message_timer = 60
                # sfx: wave_clear.wav
                return 10.0 # Wave clear reward
        return 0.0
        
    def _spawn_wave(self):
        num_zombies = self.ZOMBIE_BASE_COUNT + (self.wave - 1) * self.ZOMBIE_COUNT_INCREASE
        for _ in range(num_zombies):
            # Spawn on edges
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = [self.np_random.random() * self.W, -self.ZOMBIE_SIZE]
            elif edge == 1: # Bottom
                pos = [self.np_random.random() * self.W, self.H + self.ZOMBIE_SIZE]
            elif edge == 2: # Left
                pos = [-self.ZOMBIE_SIZE, self.np_random.random() * self.H]
            else: # Right
                pos = [self.W + self.ZOMBIE_SIZE, self.np_random.random() * self.H]

            self.zombies.append({
                "pos": pos,
                "vel": [0, 0],
                "ai_timer": self.np_random.integers(0, self.ZOMBIE_AI_UPDATE_RATE),
                "anim_timer": self.np_random.integers(0, 20)
            })
            
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.W, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.W, y))

        # Draw zombies
        for z in self.zombies:
            bob = math.sin(z["anim_timer"] / 20 * math.pi * 2) * 2
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, 
                             (int(z["pos"][0] - self.ZOMBIE_SIZE / 2),
                              int(z["pos"][1] - self.ZOMBIE_SIZE / 2 + bob),
                              self.ZOMBIE_SIZE, self.ZOMBIE_SIZE))

        # Draw bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, 
                               (int(b["pos"][0]), int(b["pos"][1])), 
                               self.BULLET_SIZE / 2)
        
        # Draw particles
        for p in self.particles:
            size = max(1, p["life"] / 6)
            pygame.draw.rect(self.screen, p["color"], 
                             (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # Draw player
        player_color = self.COLOR_PLAYER
        if self.player_hit_timer > 0 and (self.steps // 3) % 2 == 0:
            player_color = self.COLOR_PLAYER_FLASH
        
        # Player as a triangle pointing in the current direction
        p_x, p_y = int(self.player_pos[0]), int(self.player_pos[1])
        angle = self.player_direction_idx * math.pi / 2 - math.pi / 2 # Convert index to angle
        s = self.PLAYER_SIZE * 0.8
        p1 = (p_x + s * math.cos(angle), p_y + s * math.sin(angle))
        p2 = (p_x + s * 0.5 * math.cos(angle + 2.2), p_y + s * 0.5 * math.sin(angle + 2.2))
        p3 = (p_x + s * 0.5 * math.cos(angle - 2.2), p_y + s * 0.5 * math.sin(angle - 2.2))
        
        # Ensure points are integers for gfxdraw
        points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        
        pygame.gfxdraw.aapolygon(self.screen, points, player_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, player_color)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 15))

        # Wave Text
        wave_surf = self.FONT_UI.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.W - wave_surf.get_width() - 10, 10))

        # Score Text
        score_surf = self.FONT_UI.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.W / 2 - score_surf.get_width() / 2, self.H - 30))

        # Game Over / Win Message
        if self.message_timer > 0:
            msg_surf = self.FONT_MSG.render(self.game_message, True, self.COLOR_MSG)
            self.screen.blit(msg_surf, (self.W / 2 - msg_surf.get_width() / 2, self.H / 2 - msg_surf.get_height() / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "zombies_remaining": len(self.zombies)
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (after a reset)
        obs, _ = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert obs.dtype == np.uint8
        
        # Test reset again
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Requires a display. Will not work in a purely headless environment.
    is_headless = "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy"
    
    if not is_headless:
        # Re-initialize pygame for display
        pygame.quit()
        pygame.init()
        pygame.font.init()

        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.W, env.H))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        # Remove the validation call from the interactive block
        # as it's meant for headless verification.
        # env.validate_implementation() 
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                pygame.time.wait(2000) # Pause before restarting
                
            clock.tick(30) # Run at 30 FPS
            
        env.close()
    else:
        # If running in a headless environment, just validate
        env = GameEnv()
        env.validate_implementation()
        env.close()