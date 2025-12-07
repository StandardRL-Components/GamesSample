
# Generated: 2025-08-28T00:10:11.656242
# Source Brief: brief_03706.md
# Brief Index: 3706

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←/→ to jump to the nearest platform. Hold Shift or Space while jumping for a risky long jump."
    )

    game_description = (
        "Hop across procedurally generated platforms, collecting coins to reach a target score. Longer jumps are risky but offer bonus points."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 20
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150, 100)
        self.COLOR_PLATFORM = (200, 200, 220)
        self.COLOR_PLATFORM_OUTLINE = (100, 100, 120)
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_COIN_OUTLINE = (200, 160, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.PARTICLE_COLORS = [(255, 255, 100), (255, 215, 0), (250, 250, 250)]
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Game State ---
        self.game_state = {}
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game_state = {
            "steps": 0,
            "score": 0,
            "terminated": False,
            "player_pos": np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=float),
            "player_last_pos": np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=float),
            "player_size": np.array([20, 20]),
            "platforms": [],
            "coins": [],
            "particles": [],
            "camera_y": self.HEIGHT - 100,
            "base_gap": 20,
            "last_jump_distances": deque(maxlen=3),
            "last_jump_dist": 0.0,
        }

        # Create initial platform under the player
        start_platform = pygame.Rect(
            self.game_state["player_pos"][0] - 40,
            self.game_state["player_pos"][1] + self.game_state["player_size"][1],
            80, 20
        )
        self.game_state["platforms"].append(start_platform)
        
        # Generate the initial world
        for _ in range(15):
            self._generate_platform()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_state["terminated"]:
            # If the game is over, do nothing until reset
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.game_state["steps"] += 1
        reward = 0
        
        is_jump_action = movement in [3, 4] # 3=left, 4=right
        
        if is_jump_action:
            direction = "left" if movement == 3 else "right"
            is_risky = space_held or shift_held
            
            # Store position for drawing jump arc
            self.game_state["player_last_pos"] = self.game_state["player_pos"].copy()
            
            # Perform the jump and get results
            jump_result = self._handle_jump(direction, is_risky)
            reward = self._calculate_reward(jump_result)
            
            if jump_result["landed"]:
                if jump_result["coin_collected"]:
                    self.game_state["score"] += 1
                    # Increase difficulty every 5 coins
                    if self.game_state["score"] > 0 and self.game_state["score"] % 5 == 0:
                        self.game_state["base_gap"] = min(100, self.game_state["base_gap"] + 1)
            else: # Fell
                self.game_state["terminated"] = True
                
        # Update particles and camera regardless of action
        self._update_particles()
        self._update_camera()

        # Check for termination conditions
        if not self.game_state["terminated"]:
             if self.game_state["score"] >= self.WIN_SCORE:
                 reward += 100
                 self.game_state["terminated"] = True
             elif self.game_state["steps"] >= self.MAX_STEPS:
                 self.game_state["terminated"] = True

        return (
            self._get_observation(),
            reward,
            self.game_state["terminated"],
            False,
            self._get_info()
        )

    def _handle_jump(self, direction, is_risky):
        player_rect = pygame.Rect(*self.game_state["player_pos"], *self.game_state["player_size"])
        current_platform = self._get_current_platform()
        
        if not current_platform: # Should not happen in normal gameplay
            return {"landed": False, "fell": True, "coin_collected": False, "jump_dist": 0}

        # Find potential target platforms
        targets = []
        for p in self.game_state["platforms"]:
            # Must be above or slightly below current platform
            if p.centery <= current_platform.centery + 20:
                if direction == "left" and p.centerx < current_platform.centerx:
                    targets.append(p)
                elif direction == "right" and p.centerx > current_platform.centerx:
                    targets.append(p)

        if not targets:
            return self._fall()

        # Sort targets by distance from player
        targets.sort(key=lambda p: abs(p.centerx - player_rect.centerx))
        
        target_platform = targets[-1] if is_risky else targets[0]
        
        # Calculate jump distance
        jump_dist = math.hypot(target_platform.centerx - player_rect.centerx, target_platform.top - player_rect.bottom)
        
        # Update player position
        self.game_state["player_pos"][0] = target_platform.centerx - self.game_state["player_size"][0] / 2
        self.game_state["player_pos"][1] = target_platform.top - self.game_state["player_size"][1]
        
        # Spawn jump trail particles
        self._spawn_jump_arc_particles(self.game_state["player_last_pos"], self.game_state["player_pos"], 20)
        
        # Spawn landing particles
        self._spawn_particle_burst(
            (target_platform.centerx, target_platform.top),
            count=15, speed_min=1, speed_max=4
        )
        
        # Check for coin collection
        coin_collected = False
        for coin in self.game_state["coins"]:
            if coin.colliderect(pygame.Rect(*self.game_state["player_pos"], *self.game_state["player_size"])):
                self.game_state["coins"].remove(coin)
                coin_collected = True
                # Coin collection sound placeholder
                # Coin collection particle burst
                self._spawn_particle_burst(
                    coin.center, count=20, speed_min=2, speed_max=5, color=self.COLOR_COIN
                )
                break
        
        # Generate a new platform to replace the one we left from
        self._generate_platform()

        return {"landed": True, "fell": False, "coin_collected": coin_collected, "jump_dist": jump_dist}

    def _fall(self):
        # Player falls off screen
        self.game_state["player_pos"][1] += 100 # Visually drop
        self._spawn_particle_burst(
            self.game_state["player_pos"], count=10, speed_min=1, speed_max=3
        )
        return {"landed": False, "fell": True, "coin_collected": False, "jump_dist": 0}

    def _calculate_reward(self, jump_result):
        if jump_result["fell"]:
            return -100.0

        reward = 0.0
        
        if jump_result["landed"]:
            reward += 0.1 # Reward for successful landing

        if jump_result["coin_collected"]:
            reward += 1.0

        jump_dist = jump_result["jump_dist"]
        if jump_dist > 0:
            # Reward for risky jumps
            if len(self.game_state["last_jump_distances"]) > 0:
                avg_dist = sum(self.game_state["last_jump_distances"]) / len(self.game_state["last_jump_distances"])
                if jump_dist > avg_dist:
                    reward += 0.5
                if jump_dist < avg_dist * 0.75:
                    reward -= 0.2
            
            # Penalize jumping shorter than previous jump
            if jump_dist < self.game_state["last_jump_dist"]:
                reward -= 0.1

            self.game_state["last_jump_distances"].append(jump_dist)
            self.game_state["last_jump_dist"] = jump_dist
            
        return reward

    def _get_current_platform(self):
        player_bottom_center = (
            self.game_state["player_pos"][0] + self.game_state["player_size"][0] / 2,
            self.game_state["player_pos"][1] + self.game_state["player_size"][1]
        )
        for p in self.game_state["platforms"]:
            if p.collidepoint(player_bottom_center) or p.collidepoint((player_bottom_center[0], player_bottom_center[1] + 1)):
                return p
        return None

    def _generate_platform(self):
        # Remove platforms far below the camera
        self.game_state["platforms"] = [
            p for p in self.game_state["platforms"] if p.top > self.game_state["camera_y"] - 100
        ]
        
        # Find the highest platform to build upon
        if not self.game_state["platforms"]: return
        
        highest_platform = min(self.game_state["platforms"], key=lambda p: p.top)
        
        # Generate new platform
        width = self.np_random.integers(50, 120)
        height = 20
        
        dx = self.np_random.integers(
            self.game_state["base_gap"] + 40, self.game_state["base_gap"] + 150
        )
        dy = self.np_random.integers(-80, 50)
        
        direction = self.np_random.choice([-1, 1])
        
        new_x = highest_platform.centerx + direction * dx
        new_y = highest_platform.top + dy
        
        # Clamp to screen bounds
        new_x = np.clip(new_x, width / 2, self.WIDTH - width / 2)
        
        new_platform = pygame.Rect(new_x - width / 2, new_y, width, height)
        
        # Ensure it doesn't overlap with existing platforms
        if not any(p.colliderect(new_platform.inflate(20, 20)) for p in self.game_state["platforms"]):
            self.game_state["platforms"].append(new_platform)
            # Chance to spawn a coin
            if self.np_random.random() < 0.3:
                coin_radius = 8
                coin_rect = pygame.Rect(new_platform.centerx - coin_radius, new_platform.top - coin_radius*2 - 5, coin_radius*2, coin_radius*2)
                self.game_state["coins"].append(coin_rect)

    def _update_camera(self):
        target_camera_y = self.game_state["player_pos"][1] - self.HEIGHT * 0.6
        self.game_state["camera_y"] += (target_camera_y - self.game_state["camera_y"]) * 0.1

    def _update_particles(self):
        updated_particles = []
        for p in self.game_state["particles"]:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                updated_particles.append(p)
        self.game_state["particles"] = updated_particles

    def _spawn_particle_burst(self, pos, count, speed_min, speed_max, color=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_min, speed_max)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            particle_color = color if color is not None else random.choice(self.PARTICLE_COLORS)
            self.game_state["particles"].append({
                'pos': np.array(pos, dtype=float), 'vel': vel,
                'lifespan': lifespan, 'color': particle_color
            })

    def _spawn_jump_arc_particles(self, start_pos, end_pos, num_particles):
        start_vec = np.array(start_pos)
        end_vec = np.array(end_pos)
        midpoint = (start_vec + end_vec) / 2
        midpoint[1] -= 60  # Arc height

        for i in range(num_particles):
            t = i / (num_particles - 1)
            # Quadratic Bezier curve for the arc
            pos = (1-t)**2 * start_vec + 2*(1-t)*t * midpoint + t**2 * end_vec
            
            lifespan = self.np_random.integers(10, 20)
            vel = self.np_random.uniform(-0.5, 0.5, size=2)
            self.game_state["particles"].append({
                'pos': pos, 'vel': vel, 'lifespan': lifespan,
                'color': random.choice(self.PARTICLE_COLORS)
            })

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        cam_y = self.game_state["camera_y"]

        # --- Render Game Elements ---
        # Particles
        for p in self.game_state["particles"]:
            size = max(1, p['lifespan'] / 5)
            pos = (int(p['pos'][0]), int(p['pos'][1] - cam_y))
            pygame.draw.circle(self.screen, p['color'], pos, int(size))
        
        # Platforms
        for p in self.game_state["platforms"]:
            p_screen = p.move(0, -cam_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p_screen, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, p_screen, 1, border_radius=3)
        
        # Coins
        for c in self.game_state["coins"]:
            c_screen = c.move(0, -cam_y)
            pygame.gfxdraw.filled_circle(self.screen, c_screen.centerx, c_screen.centery, c.width // 2, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, c_screen.centerx, c_screen.centery, c.width // 2, self.COLOR_COIN_OUTLINE)

        # Player
        px, py = self.game_state["player_pos"]
        pw, ph = self.game_state["player_size"]
        player_rect_screen = pygame.Rect(int(px), int(py - cam_y), int(pw), int(ph))
        
        # Glow effect for player
        glow_surf = pygame.Surface((pw*2, ph*2), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=6)
        self.screen.blit(glow_surf, (player_rect_screen.x - pw/2, player_rect_screen.y - ph/2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_screen, border_radius=3)

        # --- Render UI ---
        score_text = self.font_large.render(f"SCORE: {self.game_state['score']}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"STEPS: {self.game_state['steps']}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

        if self.game_state["terminated"]:
            end_text_str = "YOU WON!" if self.game_state["score"] >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_PLAYER)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.game_state["score"],
            "steps": self.game_state["steps"],
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy" or "windows"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Platform Hopper")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        
        # Only take an action if a key is pressed
        should_step = False
        if keys[pygame.K_LEFT]:
            action[0] = 3
            should_step = True
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            should_step = True
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since it's turn-based, wait for next input
        # A small delay to prevent ultra-fast inputs
        clock.tick(10)

    env.close()
    pygame.quit()