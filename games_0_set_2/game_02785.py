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


# Set SDL_VIDEODRIVER to "dummy" to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to squash nearby bugs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hunt down and squash 50 bugs in a top-down arcade environment before your health runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game constants
        self.COLOR_BG = (40, 80, 40)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 150, 150)
        self.COLOR_SQUASH_EFFECT = (255, 255, 255)
        self.COLOR_SPLAT = (200, 200, 200)
        self.COLOR_HEALTH_FG = (60, 220, 60)
        self.COLOR_HEALTH_BG = (180, 40, 40)
        self.COLOR_TEXT = (255, 255, 255)

        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.BUG_SIZE = 8
        self.NUM_BUGS = 15
        self.SQUASH_RADIUS = 50
        self.SQUASH_DURATION = 3
        self.SQUASH_COOLDOWN = 10
        self.MAX_HEALTH = 100
        self.WIN_SCORE = 50
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.bugs = None
        self.splats = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.bugs_squashed = None
        self.base_bug_speed = None
        self.squash_timer = None
        self.squash_cooldown_timer = None
        self.last_space_held = None

        # self.reset() is called here in the original code, but it's better to
        # let the user/runner call it explicitly for the first time.
        # However, to maintain the original structure's side-effect of a fully
        # initialized object, we can call it. But the fix is in reset() itself.
        # self.reset() 
        # For the provided test harness, calling reset in init is expected.
        self.reset()

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all state variables before any methods that use them are called.
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_health = self.MAX_HEALTH
        self.score = 0
        self.steps = 0
        self.bugs_squashed = 0
        self.game_over = False
        self.base_bug_speed = 0.5  # FIX: Initialize this before _spawn_bug is called.
        self.squash_timer = 0
        self.squash_cooldown_timer = 0
        self.last_space_held = False
        self.splats = []
        
        self.bugs = []
        for _ in range(self.NUM_BUGS):
            self._spawn_bug()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty

        # 1. Handle Input & Player State
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.squash_cooldown_timer == 0:
            self.squash_timer = self.SQUASH_DURATION
            self.squash_cooldown_timer = self.SQUASH_COOLDOWN
            # sfx: whoosh

        self.last_space_held = space_held

        # 2. Update Game World
        self._update_splats()
        self._update_bugs()

        # 3. Handle Interactions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Squash interaction
        if self.squash_timer > 0:
            bugs_to_remove = []
            for bug in self.bugs:
                dist = math.hypot(self.player_pos[0] - bug['pos'][0], self.player_pos[1] - bug['pos'][1])
                if dist < self.SQUASH_RADIUS:
                    bugs_to_remove.append(bug)
                    reward += 1.0 # sfx: splat_sound
                    if bug['age'] <= 5: reward += 0.1 # Speed bonus
                    self.score += 1
                    self.bugs_squashed += 1
                    self.splats.append({'pos': bug['pos'][:], 'radius': 15, 'life': 10})
            self.bugs = [b for b in self.bugs if b not in bugs_to_remove]

        # Player-bug collision
        bugs_to_remove = []
        for bug in self.bugs:
            bug_rect = pygame.Rect(bug['pos'][0] - self.BUG_SIZE/2, bug['pos'][1] - self.BUG_SIZE/2, self.BUG_SIZE, self.BUG_SIZE)
            if player_rect.colliderect(bug_rect):
                bugs_to_remove.append(bug)
                self.player_health -= 10
                reward -= 1.0 # sfx: player_hit_sound
                self.splats.append({'pos': bug['pos'][:], 'radius': 15, 'life': 10, 'color': self.COLOR_PLAYER})
        self.bugs = [b for b in self.bugs if b not in bugs_to_remove]
        
        # 4. Update Game State
        while len(self.bugs) < self.NUM_BUGS:
            self._spawn_bug()

        self.base_bug_speed = 0.5 + (self.bugs_squashed // 5) * 0.05
        
        self.squash_timer = max(0, self.squash_timer - 1)
        self.squash_cooldown_timer = max(0, self.squash_cooldown_timer - 1)
        self.steps += 1
        
        # 5. Check Termination
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 100
            terminated = True
            # sfx: game_over_lose
        elif self.bugs_squashed >= self.WIN_SCORE:
            reward += 100
            terminated = True
            # sfx: game_over_win
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_bug(self):
        while True:
            pos = [random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)]
            dist_to_player = math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1])
            if dist_to_player > self.SQUASH_RADIUS * 2:
                break
        
        angle = random.uniform(0, 2 * math.pi)
        speed = self.base_bug_speed + random.uniform(-0.2, 0.2)
        velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
        color = random.choice([(180, 255, 180), (255, 255, 150), (150, 200, 255)])

        self.bugs.append({'pos': pos, 'vel': velocity, 'color': color, 'age': 0})

    def _update_bugs(self):
        for bug in self.bugs:
            bug['pos'][0] += bug['vel'][0]
            bug['pos'][1] += bug['vel'][1]
            bug['age'] += 1

            # Wrap around screen
            if bug['pos'][0] < 0: bug['pos'][0] = self.WIDTH
            if bug['pos'][0] > self.WIDTH: bug['pos'][0] = 0
            if bug['pos'][1] < 0: bug['pos'][1] = self.HEIGHT
            if bug['pos'][1] > self.HEIGHT: bug['pos'][1] = 0

            # Slightly alter course
            if random.random() < 0.05:
                angle = math.atan2(bug['vel'][1], bug['vel'][0]) + random.uniform(-0.5, 0.5)
                speed = math.hypot(bug['vel'][0], bug['vel'][1])
                bug['vel'] = [math.cos(angle) * speed, math.sin(angle) * speed]

    def _update_splats(self):
        self.splats = [s for s in self.splats if s['life'] > 0]
        for s in self.splats:
            s['life'] -= 1
            s['radius'] += 0.5

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render splats (background)
        for s in self.splats:
            alpha = int(255 * (s['life'] / 10))
            color = s.get('color', self.COLOR_SPLAT)
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((s['radius']*2, s['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(s['radius']), int(s['radius']), int(s['radius']), (*color, alpha))
            self.screen.blit(temp_surf, (int(s['pos'][0] - s['radius']), int(s['pos'][1] - s['radius'])))

        # Render bugs
        for bug in self.bugs:
            pos = (int(bug['pos'][0]), int(bug['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BUG_SIZE, bug['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BUG_SIZE, bug['color'])

        # Render player
        player_center = (int(self.player_pos[0]), int(self.player_pos[1]))
        player_rect = pygame.Rect(player_center[0] - self.PLAYER_SIZE/2, player_center[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_size = self.PLAYER_SIZE + 10
        glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
        glow_rect.center = player_center
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (0, 0, glow_size, glow_size))
        self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Render squash effect
        if self.squash_timer > 0:
            alpha = int(255 * (self.squash_timer / self.SQUASH_DURATION))
            radius = self.SQUASH_RADIUS
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.COLOR_SQUASH_EFFECT, alpha // 4))
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (*self.COLOR_SQUASH_EFFECT, alpha))
            self.screen.blit(temp_surf, (player_center[0] - radius, player_center[1] - radius))
        
        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, bar_height))
        
        health_color = self.COLOR_HEALTH_FG
        if health_ratio < 0.5:
            health_color = (255, 255, 0) # Yellow
        if health_ratio < 0.25:
            health_color = (255, 0, 0) # Red

        pygame.draw.rect(self.screen, health_color, (10, 10, int(bar_width * health_ratio), bar_height))
        
        # Score
        score_text = self.font_large.render(f"{self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            msg = "YOU WIN!" if self.bugs_squashed >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "bugs_squashed": self.bugs_squashed,
        }
        
    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # To run and play the game
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard to MultiDiscrete action space ---
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
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    pygame.quit()
    print(f"Game Over! Final Info: {info}")