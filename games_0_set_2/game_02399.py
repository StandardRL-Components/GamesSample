
# Generated: 2025-08-28T04:42:38.310132
# Source Brief: brief_02399.md
# Brief Index: 2399

        
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
        "Controls: Arrow keys to move. Hold Space to shoot in your last movement direction. Survive the horde!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 10 minutes against an ever-growing zombie horde. Collect supplies to heal and shoot to clear your path."
    )

    # Frames auto-advance at a fixed rate.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.ARENA_MARGIN = 20
        self.MAX_STEPS = 6000  # 10 minutes at 10 steps/sec (see clock logic in step)
        
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_ARENA = (40, 40, 50)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_PLAYER_GLOW = (0, 255, 127, 50)
        self.COLOR_ZOMBIE = (255, 69, 0) # Red-Orange
        self.COLOR_ZOMBIE_HIT = (255, 255, 255)
        self.COLOR_SUPPLY = (30, 144, 255) # Dodger Blue
        self.COLOR_BULLET = (255, 255, 0) # Yellow
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (46, 204, 113)
        self.COLOR_HEALTH_BAR_BG = (192, 57, 43)

        # Entity properties
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 4.0
        self.ZOMBIE_SIZE = 8
        self.SUPPLY_SIZE = 7
        self.BULLET_SIZE = 3
        self.BULLET_SPEED = 8.0
        self.SHOOT_COOLDOWN_MAX = 5 # steps
        self.INITIAL_ZOMBIE_COUNT = 100
        self.INITIAL_SUPPLY_COUNT = 50
        self.SUPPLY_RESPAWN_TIME = 300 # steps
        self.SUPPLY_HEAL_AMOUNT = 20
        self.ZOMBIE_DAMAGE = 10
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.last_move_direction = None
        self.shoot_cooldown = 0
        self.zombies = []
        self.supplies = []
        self.bullets = []
        self.particles = []
        self.zombie_speed = 0
        self.player_invulnerability_timer = 0
        
        # --- Finalization ---
        self.reset()
        self.validate_implementation(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_health = 100
        self.player_invulnerability_timer = 0
        self.last_move_direction = (0, -1)  # Default to shooting up
        self.shoot_cooldown = 0

        self.zombie_speed = 0.5
        self.zombies = []
        for _ in range(self.INITIAL_ZOMBIE_COUNT):
            self.zombies.append(self._create_zombie())

        self.supplies = []
        for _ in range(self.INITIAL_SUPPLY_COUNT):
            self.supplies.append(self._create_supply())
        
        self.bullets = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        # The game runs at a logical 10 FPS for RL, but renders smoothly.
        # This is a simple way to manage game speed with auto_advance.
        self.clock.tick(30) # For visual smoothness, though logic is step-based

        reward = 0.1  # Survival reward per step

        if not self.game_over:
            # --- Handle Input & Cooldowns ---
            self._handle_input(action)
            if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
            if self.player_invulnerability_timer > 0: self.player_invulnerability_timer -= 1
            
            # --- Update Game State ---
            self._update_bullets()
            self._update_zombies()
            self._update_supplies()
            self._update_particles()

            # --- Handle Collisions & Events ---
            collision_rewards = self._handle_collisions()
            reward += collision_rewards

            # --- Update Progression ---
            self.steps += 1
            if self.steps > 0 and self.steps % 600 == 0:
                self.zombie_speed += 0.01

        # --- Check Termination ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health > 0:
                reward += 100.0  # Victory bonus

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_vector = [0, 0]
        if movement == 1:  # Up
            move_vector[1] -= 1
        elif movement == 2:  # Down
            move_vector[1] += 1
        elif movement == 3:  # Left
            move_vector[0] -= 1
        elif movement == 4:  # Right
            move_vector[0] += 1
        
        if movement != 0:
            self.last_move_direction = tuple(move_vector)
            self.player_pos[0] += move_vector[0] * self.PLAYER_SPEED
            self.player_pos[1] += move_vector[1] * self.PLAYER_SPEED

        # Clamp player position
        self.player_pos[0] = max(self.ARENA_MARGIN + self.PLAYER_SIZE, min(self.WIDTH - self.ARENA_MARGIN - self.PLAYER_SIZE, self.player_pos[0]))
        self.player_pos[1] = max(self.ARENA_MARGIN + self.PLAYER_SIZE, min(self.HEIGHT - self.ARENA_MARGIN - self.PLAYER_SIZE, self.player_pos[1]))

        if space_held and self.shoot_cooldown == 0:
            self._fire_bullet()
            
    def _fire_bullet(self):
        # sfx: player_shoot
        self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX
        bullet_pos = list(self.player_pos)
        self.bullets.append({"pos": bullet_pos, "dir": self.last_move_direction})

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet["pos"][0] += bullet["dir"][0] * self.BULLET_SPEED
            bullet["pos"][1] += bullet["dir"][1] * self.BULLET_SPEED
            if not (0 < bullet["pos"][0] < self.WIDTH and 0 < bullet["pos"][1] < self.HEIGHT):
                self.bullets.remove(bullet)

    def _update_zombies(self):
        for z in self.zombies:
            if z['hit_timer'] > 0:
                z['hit_timer'] -= 1
            
            # Move towards player
            dx = self.player_pos[0] - z["pos"][0]
            dy = self.player_pos[1] - z["pos"][1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                dx, dy = dx / dist, dy / dist
                z["pos"][0] += dx * self.zombie_speed
                z["pos"][1] += dy * self.zombie_speed

    def _update_supplies(self):
        for s in self.supplies:
            if s['respawn_timer'] > 0:
                s['respawn_timer'] -= 1
                if s['respawn_timer'] == 0:
                    s['pos'] = self._get_random_pos()

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][0] *= 0.95 # friction
                p['vel'][1] *= 0.95
                p['size'] = max(0, p['size'] * 0.9)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Zombies
        for bullet in self.bullets[:]:
            for zombie in self.zombies[:]:
                if zombie['hit_timer'] > 0: continue # Already hit this frame
                dist = math.hypot(bullet["pos"][0] - zombie["pos"][0], bullet["pos"][1] - zombie["pos"][1])
                if dist < self.ZOMBIE_SIZE + self.BULLET_SIZE:
                    # sfx: zombie_die
                    self.score += 10
                    reward += 10.0
                    self._create_particles(zombie["pos"], self.COLOR_ZOMBIE, 10)
                    self.zombies.remove(zombie)
                    self.zombies.append(self._create_zombie()) # Respawn immediately
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    break

        # Player vs Zombies
        if self.player_invulnerability_timer == 0:
            for zombie in self.zombies:
                dist = math.hypot(self.player_pos[0] - zombie["pos"][0], self.player_pos[1] - zombie["pos"][1])
                if dist < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                    # sfx: player_hurt
                    self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                    self.player_invulnerability_timer = 15 # 0.5s invulnerability
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 5)
                    break # Only take damage from one zombie per frame

        # Player vs Supplies
        for supply in self.supplies:
            if supply['respawn_timer'] > 0: continue
            dist = math.hypot(self.player_pos[0] - supply["pos"][0], self.player_pos[1] - supply["pos"][1])
            if dist < self.PLAYER_SIZE + self.SUPPLY_SIZE:
                # sfx: supply_pickup
                self.player_health = min(100, self.player_health + self.SUPPLY_HEAL_AMOUNT)
                self.score += 5
                reward += 5.0
                supply['respawn_timer'] = self.SUPPLY_RESPAWN_TIME
                self._create_particles(supply['pos'], self.COLOR_SUPPLY, 8)
        
        return reward

    def _get_random_pos(self):
        return [
            self.np_random.uniform(self.ARENA_MARGIN, self.WIDTH - self.ARENA_MARGIN),
            self.np_random.uniform(self.ARENA_MARGIN, self.HEIGHT - self.ARENA_MARGIN)
        ]

    def _create_zombie(self):
        pos = list(self._get_random_pos())
        # Ensure zombies don't spawn on the player
        while math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]) < 100:
            pos = self._get_random_pos()
        return {"pos": pos, "hit_timer": 0}

    def _create_supply(self):
        return {"pos": self._get_random_pos(), "respawn_timer": 0}
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })

    def _get_observation(self):
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (self.ARENA_MARGIN, self.ARENA_MARGIN, self.WIDTH - 2 * self.ARENA_MARGIN, self.HEIGHT - 2 * self.ARENA_MARGIN))

        # --- Render Game Elements ---
        for s in self.supplies:
            if s['respawn_timer'] == 0:
                pygame.draw.rect(self.screen, self.COLOR_SUPPLY, (int(s['pos'][0] - self.SUPPLY_SIZE), int(s['pos'][1] - self.SUPPLY_SIZE), self.SUPPLY_SIZE * 2, self.SUPPLY_SIZE * 2))

        for z in self.zombies:
            color = self.COLOR_ZOMBIE_HIT if z['hit_timer'] > 0 else self.COLOR_ZOMBIE
            pygame.gfxdraw.filled_circle(self.screen, int(z['pos'][0]), int(z['pos'][1]), self.ZOMBIE_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int(z['pos'][0]), int(z['pos'][1]), self.ZOMBIE_SIZE, color)

        for b in self.bullets:
            pygame.gfxdraw.filled_circle(self.screen, int(b['pos'][0]), int(b['pos'][1]), self.BULLET_SIZE, self.COLOR_BULLET)

        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (int(p['size']), int(p['size'])), int(p['size']))
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Player (with invulnerability flash)
        if self.player_invulnerability_timer % 4 < 2:
            # Glow effect
            glow_radius = int(self.PLAYER_SIZE * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (int(self.player_pos[0] - glow_radius), int(self.player_pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
            # Player circle
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.COLOR_PLAYER)
        
        # --- Render UI ---
        # Health Bar
        health_bar_width = 150
        health_ratio = self.player_health / 100
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(health_bar_width * health_ratio), 20))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        minutes = time_left // 600
        seconds = (time_left % 600) // 10
        timer_text = f"TIME: {minutes:02}:{seconds:02}"
        text_surface = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        text_surface = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH // 2 - text_surface.get_width() // 2, 10))

        # Game Over Text
        if self.game_over:
            msg = "VICTORY!" if self.player_health > 0 else "GAME OVER"
            text_surface = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "zombie_speed": self.zombie_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self, test_instance):
        # Test action space
        assert test_instance.action_space.shape == (3,)
        assert test_instance.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = test_instance._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = test_instance.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = test_instance.action_space.sample()
        obs, reward, term, trunc, info = test_instance.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
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

        # --- Pygame Rendering ---
        # The observation is already the rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(60) # Run the display loop at 60 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()