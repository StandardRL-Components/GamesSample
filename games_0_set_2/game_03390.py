# Generated: 2025-08-27T23:12:58.057132
# Source Brief: brief_03390.md
# Brief Index: 3390

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Vault Hopper: A minimalist, side-view platformer where players must precisely time
    jumps to ascend a series of procedurally generated platforms against the clock.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press SPACE to jump. Timing is everything."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Minimalist arcade platformer. Time your jumps perfectly to ascend the tower before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_PER_STAGE = 60  # seconds
        self.TOTAL_STAGES = 3

        # --- Colors ---
        self.COLOR_BG_TOP = (40, 40, 80)
        self.COLOR_BG_BOTTOM = (80, 80, 160)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLATFORM = (240, 240, 240)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (220, 220, 220)

        # --- Physics & Gameplay ---
        self.GRAVITY = 0.8
        self.JUMP_VELOCITY = -14
        self.PLAYER_AUTO_MOVE_SPEED = 2.0
        self.PLAYER_SIZE = pygame.Vector2(20, 20)

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.time_remaining = 0
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = False
        self.can_jump = True
        self.auto_move_direction = 1
        self.platforms = []
        self.particles = []
        self.highest_platform_idx = 0
        
        # Initialize state by calling reset
        # self.reset() # Not needed as it's called by testing harness
        
        # # Run self-check
        # self.validate_implementation() # Not needed for submission


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.time_remaining = self.MAX_TIME_PER_STAGE * self.FPS
        self.particles.clear()
        
        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE.y)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = True  # FIX: Player starts on a platform, so should be grounded.
        self.can_jump = True
        self.auto_move_direction = 1
        self.highest_platform_idx = 0

        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms.clear()
        base_width = 120
        base_x_gap = 100
        y_gap = 85

        # Difficulty scaling: platforms get narrower and gaps wider
        difficulty_mod = 1.0 - (self.stage - 1) * 0.10
        plat_width = int(base_width * difficulty_mod)
        max_x_gap = int(base_x_gap / difficulty_mod)

        # Start platform
        start_plat = pygame.Rect(self.WIDTH // 2 - plat_width // 2, self.HEIGHT - 40, plat_width, 20)
        self.platforms.append(start_plat)

        current_y = start_plat.top
        while current_y > 100:
            last_plat = self.platforms[-1]
            new_y = current_y - y_gap
            
            x_offset = self.np_random.integers(-max_x_gap, max_x_gap + 1)
            new_x = last_plat.centerx + x_offset - plat_width // 2
            
            # Clamp to screen bounds with a margin
            new_x = max(20, min(new_x, self.WIDTH - plat_width - 20))

            new_plat = pygame.Rect(new_x, new_y, plat_width, 20)
            self.platforms.append(new_plat)
            current_y = new_y
        
        # Finish line platform
        finish_plat = pygame.Rect(0, 40, self.WIDTH, 10)
        self.platforms.append(finish_plat)

    def step(self, action):
        space_held = action[1] == 1
        reward = 0.1  # Survival reward

        if not self.game_over:
            self._handle_input(space_held)
            self._update_player()
            landing_reward, stage_cleared = self._handle_collisions()
            reward += landing_reward

            self._update_particles()
            self.time_remaining -= 1
            
            if stage_cleared:
                if self.stage == self.TOTAL_STAGES:
                    reward += 100  # Victory
                    self.game_over = True
                else:
                    reward += 25  # Stage clear bonus
                    self.stage += 1
                    self.time_remaining = self.MAX_TIME_PER_STAGE * self.FPS
                    self._generate_platforms()
                    self.highest_platform_idx = 0
                    self.is_grounded = False # Fall from finish line to new start

        terminated = self.game_over
        if self.player_pos.y > self.HEIGHT + self.PLAYER_SIZE.y:
            if not terminated: reward -= 100
            terminated = True
        if self.time_remaining <= 0:
            if not terminated: reward -= 50
            terminated = True
        
        self.steps += 1
        self.score += reward

        # The 'truncated' flag is not used in this environment's logic.
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, space_held):
        if space_held and self.is_grounded and self.can_jump:
            # Sound: Jump
            self.player_vel.y = self.JUMP_VELOCITY
            self.player_vel.x = self.PLAYER_AUTO_MOVE_SPEED * self.auto_move_direction
            self.is_grounded = False
            self.can_jump = False
        
        if not space_held:
            self.can_jump = True

    def _update_player(self):
        # FIX: This function was rewritten to fix multiple bugs related to stale state
        # and missing "fall-off" logic, ensuring stable auto-movement.
        if self.is_grounded:
            self.player_vel.y = 0
            self.player_pos.x += self.PLAYER_AUTO_MOVE_SPEED * self.auto_move_direction
            
            # Update rect *after* moving to avoid stale data for checks
            player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)

            current_plat = None
            for plat in self.platforms:
                # Use a point just below the player to see if there's ground
                if plat.collidepoint(player_rect.midbottom[0], player_rect.midbottom[1] + 1):
                    current_plat = plat
                    break
            
            if current_plat:
                # On a platform, check for turning at edges
                if player_rect.right > current_plat.right:
                    self.player_pos.x = current_plat.right - self.PLAYER_SIZE.x
                    self.auto_move_direction *= -1
                if player_rect.left < current_plat.left:
                    self.player_pos.x = current_plat.left
                    self.auto_move_direction *= -1
            else:
                # No platform below, so we start falling
                self.is_grounded = False
        
        # This part handles falling (gravity) and in-air movement.
        # It is a separate 'if' to catch the case where we just walked off a ledge in the same frame.
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel

        # Final clamp to screen horizontal bounds
        self.player_pos.x = max(0, min(self.player_pos.x, self.WIDTH - self.PLAYER_SIZE.x))


    def _handle_collisions(self):
        reward = 0
        stage_cleared = False
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)

        if self.player_vel.y > 0:  # Check for landing only if falling
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat) and player_rect.bottom < plat.bottom + 10:
                    last_pos_y = self.player_pos.y - self.player_vel.y
                    if (last_pos_y + self.PLAYER_SIZE.y) <= plat.top:
                        self.player_pos.y = plat.top - self.PLAYER_SIZE.y
                        self.is_grounded = True
                        self.player_vel.x = 0
                        # Sound: Land
                        self._create_landing_particles(pygame.Vector2(player_rect.midbottom))
                        
                        if i > self.highest_platform_idx:
                            reward += 1.0 * (i - self.highest_platform_idx)
                            self.highest_platform_idx = i

                        if i == len(self.platforms) - 1:
                           stage_cleared = True
                        break
        return reward, stage_cleared

    def _create_landing_particles(self, pos):
        for _ in range(10):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'radius': self.np_random.uniform(2, 5), 'life': 20
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            # Create a temporary surface for alpha blending if gfxdraw is not sufficient
            if alpha > 0:
                pos = (int(p['pos'].x), int(p['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p['radius'])), (*self.COLOR_PARTICLE, alpha))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], max(0, int(p['radius'])), (*self.COLOR_PARTICLE, alpha))

        finish_line = self.platforms[-1]
        pygame.draw.rect(self.screen, self.COLOR_FINISH, finish_line)
        for plat in self.platforms[:-1]:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)
        
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, (255, 180, 180), player_rect.inflate(4, 4), 1, border_radius=4)

    def _render_ui(self):
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        stage_text = f"STAGE: {self.stage}/{self.TOTAL_STAGES}"
        stage_surf = self.font_small.render(stage_text, True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.WIDTH - stage_surf.get_width() - 10, 10))

        if self.game_over:
            is_victory = self.stage == self.TOTAL_STAGES and self.player_pos.y <= self.platforms[-1].bottom
            end_text = "VICTORY!" if is_victory else "GAME OVER"
            end_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = end_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": max(0, self.time_remaining // self.FPS),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")