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

    user_guide = (
        "Controls: Use ← and → to run. Press ↑ or Space to jump. Reach the green flag before time runs out!"
    )

    game_description = (
        "A minimalist pixel-art platformer. Navigate procedurally generated levels, collect gold bonuses for score, and reach the flag within the time limit."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 900  # 30 seconds at 30fps

        # Physics constants
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -12
        self.FRICTION = 0.8

        # Color palette
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (60, 80, 120)
        self.COLOR_PLATFORM = (100, 100, 110)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 150, 150, 50)
        self.COLOR_BONUS = (255, 215, 0)
        self.COLOR_BONUS_GLOW = (255, 215, 0, 60)
        self.COLOR_FLAG = (80, 220, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (30, 30, 30)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20, bold=True)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.difficulty_level = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.bonuses = []
        self.particles = []
        self.flag_rect = pygame.Rect(0, 0, 0, 0)
        self.prev_dist_to_flag = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()

        # Increase difficulty for subsequent resets
        if options and options.get("new_game", True):
            self.difficulty_level = 0
        else:
            self.difficulty_level += 1

        # Procedurally generate level
        self._generate_level()

        # Reset player state
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - 30)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.prev_dist_to_flag = abs(self.player_pos.x - self.flag_rect.centerx)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]
        space_pressed = action[1] == 1

        # --- 1. Handle Input ---
        self._handle_input(movement, space_pressed)

        # --- 2. Update Physics ---
        self._update_physics()

        # --- 3. Handle Collisions ---
        self._handle_collisions()

        # --- 4. Update Game Logic ---
        reward = self._calculate_and_apply_rewards()
        self._update_particles()

        # --- 5. Check Termination ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED

        # Jumping
        if (movement == 1 or space_pressed) and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()
            self._create_jump_particles(self.player_pos + pygame.Vector2(0, 15))

    def _update_physics(self):
        # Apply friction if no horizontal input
        if self.player_vel.x != 0 and abs(self.player_vel.x) < self.PLAYER_SPEED:
            self.player_vel.x *= self.FRICTION
            if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 15: # Terminal velocity
            self.player_vel.y = 15

        # Update position
        self.player_pos += self.player_vel
        self.player_vel.x = 0 # Reset horizontal velocity intention each frame

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        self.on_ground = False

        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Vertical collision (landing on top)
                if self.player_vel.y > 0 and player_rect.bottom > plat.top and player_rect.top < plat.top:
                    player_rect.bottom = plat.top
                    self.player_pos.y = player_rect.centery
                    if self.player_vel.y > 2: # Create landing particles on hard falls
                        # sfx: land_sound()
                        self._create_jump_particles(self.player_pos + pygame.Vector2(0, 10))
                    self.player_vel.y = 0
                    self.on_ground = True
                # Horizontal collision
                elif self.player_vel.x > 0 and player_rect.right > plat.left and player_rect.left < plat.left:
                    player_rect.right = plat.left
                    self.player_pos.x = player_rect.centerx
                    self.player_vel.x = 0
                elif self.player_vel.x < 0 and player_rect.left < plat.right and player_rect.right > plat.right:
                    player_rect.left = plat.right
                    self.player_pos.x = player_rect.centerx
                    self.player_vel.x = 0

    def _calculate_and_apply_rewards(self):
        reward = -0.01  # Small penalty for each step to encourage speed

        # Distance-based reward
        current_dist = abs(self.player_pos.x - self.flag_rect.centerx)
        if current_dist < self.prev_dist_to_flag:
            reward += 0.1
        elif current_dist > self.prev_dist_to_flag:
            reward -= 0.1
        self.prev_dist_to_flag = current_dist

        # Bonus collection
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)
        for bonus in self.bonuses[:]:
            if player_rect.colliderect(bonus):
                self.bonuses.remove(bonus)
                self.score += 25
                reward += 5
                # sfx: bonus_collect_sound()

        return reward

    def _check_termination(self):
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 10, 20, 20)

        # Win condition
        if player_rect.colliderect(self.flag_rect):
            self.score += 100
            # sfx: win_sound()
            return True

        # Loss conditions
        if self.player_pos.y > self.HEIGHT + 20: # Fell off screen
            self.score -= 100
            # sfx: fall_sound()
            return True
        if self.steps >= self.MAX_STEPS: # Time out
            return True

        return False

    def _generate_level(self):
        self.platforms.clear()
        self.bonuses.clear()

        # Difficulty parameters
        min_plat_w = max(40, 150 - self.difficulty_level * 8)
        max_plat_w = max(80, 250 - self.difficulty_level * 10)
        min_gap_x = 40
        max_gap_x = min(120, 60 + self.difficulty_level * 4)
        max_gap_y = 60

        # Create starting platform
        current_x = 40
        start_y = self.HEIGHT - 50
        start_w = 200
        self.platforms.append(pygame.Rect(current_x, start_y, start_w, 20))
        current_x += start_w

        # Create subsequent platforms until screen edge is reached
        while current_x < self.WIDTH - 100:
            gap_x = self.np_random.integers(min_gap_x, max_gap_x + 1)
            gap_y = self.np_random.integers(-max_gap_y, max_gap_y + 1)
            plat_w = self.np_random.integers(min_plat_w, max_plat_w + 1)

            next_x = current_x + gap_x
            next_y = np.clip(self.platforms[-1].y + gap_y, 100, self.HEIGHT - 50)

            new_plat = pygame.Rect(next_x, next_y, plat_w, 20)
            self.platforms.append(new_plat)

            # Chance to add a bonus item in the middle of a gap
            if self.np_random.random() < 0.4 and gap_x > 80:
                bonus_x = current_x + gap_x / 2
                bonus_y = min(self.platforms[-1].y, new_plat.y) - 30
                self.bonuses.append(pygame.Rect(bonus_x - 5, bonus_y - 5, 10, 10))

            current_x = next_x + plat_w

        # Place flag on the last platform
        last_plat = self.platforms[-1]
        self.flag_rect = pygame.Rect(last_plat.right - 30, last_plat.top - 40, 20, 40)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            # Interpolate color from top to bottom
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0,50), plat.move(2,2), border_radius=3) # Shadow

        # Render bonuses
        bob_offset = math.sin(self.steps * 0.1) * 3
        for bonus in self.bonuses:
            pygame.gfxdraw.filled_circle(self.screen, bonus.centerx, int(bonus.centery + bob_offset), 8, self.COLOR_BONUS_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, bonus.centerx, int(bonus.centery + bob_offset), 6, self.COLOR_BONUS)

        # Render flag
        pygame.draw.rect(self.screen, (200,200,200), (self.flag_rect.x - 2, self.flag_rect.y, 4, self.flag_rect.height))
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [(self.flag_rect.x, self.flag_rect.y), (self.flag_rect.x + self.flag_rect.width, self.flag_rect.y + 10), (self.flag_rect.x, self.flag_rect.y + 20)])

        # Render particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            alpha = max(0, p['life'] * 20)
            size = max(1, int(p['life'] * 0.5))
            glow_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (255,255,255, alpha//4), glow_surf.get_rect())
            self.screen.blit(glow_surf, (p['pos'].x - size*2, p['pos'].y - size*2), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, (255,255,255), (p['pos'].x - size//2, p['pos'].y - size//2, size, size))

        # Render player
        player_w, player_h = 20, 20
        # Squash and stretch animation
        if not self.on_ground:
            stretch = min(5, self.player_vel.y)
            player_h += stretch
            player_w -= stretch / 2
        elif self.player_vel.y == 0 and self.on_ground: # Just landed
             squash = min(5, abs(self.player_vel.y))
             player_h -= squash
             player_w += squash

        player_rect = pygame.Rect(0, 0, max(5, player_w), max(5, player_h))
        player_rect.center = self.player_pos

        # Player Glow
        glow_surf = pygame.Surface((player_rect.width*2, player_rect.height*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_surf.get_rect(center=player_rect.center), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_small.render(score_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (self.WIDTH - score_surf.get_width() - 18, 12))
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 20, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"TIME: {time_left / self.FPS:.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_small.render(time_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (22, 12))
        self.screen.blit(time_surf, (20, 10))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_jump_particles(self, pos):
        for _ in range(8):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(10, 21)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "difficulty": self.difficulty_level,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # The main loop is for manual play and debugging, not part of the official env
    # Re-enable the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset(seed=42)
    terminated = False
    
    # Setup display for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Platformer")
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while True:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Reset after a short delay to see the final state
            pygame.time.wait(2000)
            obs, info = env.reset(seed=43) # Use a different seed for a new level
        
        clock.tick(env.FPS)